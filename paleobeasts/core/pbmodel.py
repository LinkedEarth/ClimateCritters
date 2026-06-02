from scipy.integrate import solve_ivp
import inspect
import warnings
from types import MethodType
import numpy as np

from ..utils.solver import (euler_method, euler_maruyama_method, heun_maruyama_method,
                            milstein_method, rk4_method, Solution,
                            validate_initial_state as _validate_initial_state,
                            build_state_from_history as _build_state_from_history)
from .pboutput import PBOutput


class PBModel:
    '''The overarching model structure for Paleobeasts. 
    
    PBModel serves as the archetype/parent class for models within the signal_models directory.
    This class is not meant to be instantiated, but rather to be inherited by other classes.

    Parameter handling
    ------------------
    Models may define a ``param_values`` dict that maps parameter names to either:
    - constants (floats/ints)
    - callables (time/state/model aware)
    - objects with ``get_forcing`` (e.g., ``pb.core.Forcing``)

    Use ``get_param(name, t, state)`` inside ``dydt`` to resolve time-varying parameters.
    
    '''

    def __init__(self, forcing, variable_name, state_variables=None, non_integrated_state_vars=None,
                 diagnostic_variables=None):
        self.variable_name = variable_name
        self.forcing = forcing

        if state_variables is None:
            state_variables = []
        if non_integrated_state_vars is None:
            non_integrated_state_vars = []

        self.state_variables_names = state_variables
        self.non_integrated_state_vars = non_integrated_state_vars
        self.integrated_state_vars = [var for var in state_variables if var not in self.non_integrated_state_vars]
        self.dtypes = None
        self.state_variables = None

        if diagnostic_variables is None:
            diagnostic_variables = []
        # if 'time' not in diagnostic_variables:
        #     diagnostic_variables.append('time')
        self.diagnostic_variables = {var:[] for var in diagnostic_variables}
        self.params = ()
        self.param_values = {}

        self.t_span = None   # set at start of integrate(); useful for inspection
        self.y0 = None       # set at start of integrate(); useful for inspection
        self.time = None
        self.time_util = lambda t: t
        self.rng = None

    def __setattr__(self, name, value):
        """Keep ``param_values`` in sync when attributes are set directly.

        Without this, ``model.alpha = 2.0`` would update the attribute but leave
        ``param_values['alpha']`` stale, causing the solver to use the old value.
        The sync only fires when the name already exists in ``param_values``, so
        ordinary attribute assignments are unaffected.
        """
        object.__setattr__(self, name, value)
        param_values = self.__dict__.get('param_values', None)
        if isinstance(param_values, dict) and name in param_values:
            param_values[name] = value

    def __copy__(self):
        """Shallow copy with diagnostic variables reset to empty lists.

        A plain shallow copy would share the ``diagnostic_variables`` lists
        between the original and the copy, so appends during integration would
        corrupt both.  Resetting them here ensures a copied model starts with
        no accumulated output from the original's run.
        """
        new_obj = type(self)(self.forcing, self.variable_name)
        new_obj.__dict__.update(self.__dict__)
        new_obj.diagnostic_variables = {var: [] for var in self.diagnostic_variables}
        return new_obj

    def _resolve_param(self, param, t, state):
        """Turn a raw parameter value into a number at the current time and state.

        Parameters can be stored as constants, callables, or Forcing objects.
        This method is the single place that handles all three cases, so that
        ``get_param`` and any future callers don't need to repeat the type logic.
        It is private because callers should always go through ``get_param``
        rather than resolving values directly.
        """
        if param is None:
            return None
        if hasattr(param, 'get_forcing'):
            return param.get_forcing(self.time_util(t))
        if callable(param):
            return self._dispatch_callable(param, t, state)
        return param

    def _dispatch_callable(self, param, t, state):
        """Call a callable parameter with the right arguments for its signature.

        Callable parameters can request different amounts of context: time only,
        time and state, or time, state, and the model instance.  Inspecting
        arity here means the caller (``_resolve_param``) doesn't need to know
        which variant a given callable uses — it just passes everything and lets
        this method sort it out.  The first-argument name check enforces the
        contract defined in ``contracts/signal_model_contract.md``.
        """
        try:
            sig = inspect.signature(param)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Parameter callable must be inspectable. "
                "Supported signatures: (t), (t, state), (t, state, model)."
            ) from exc

        if any(p.kind == inspect.Parameter.VAR_POSITIONAL
               for p in sig.parameters.values()):
            raise TypeError("Parameter callables may not use *args.")

        positional = [
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        n = len(positional)
        if n not in (1, 2, 3) or positional[0].name.lower() not in ('t', 'time'):
            raise TypeError(
                "Parameter callable must have signature (t), (t, state), or "
                "(t, state, model) where the first argument is named 't' or 'time'."
            )

        if n == 1:
            return param(t)
        if n == 2:
            return param(t, state)
        return param(t, state, self)



    def resolve_forcing(self, t, default=0.0):
        """Evaluate the model's external forcing at time t.

        If no forcing object is attached (``self.forcing is None``), returns
        ``default``, which the caller supplies — a fallback parameter value, a
        zero vector, or a computed internal term.  Keeping the fallback out of
        this method preserves the distinction between forcings (external drivers)
        and parameters (intrinsic model properties).
        """
        if self.forcing is None:
            return default
        return self.forcing.get_forcing(self.time_util(t))

    def get_param_value(self, name, t, state):
        """Resolve a named parameter to its value at the current time and state.

        This is the standard way to access parameters inside ``dydt``.  It looks
        up the value from ``param_values`` by name and delegates resolution to
        ``_resolve_param``, so callers in ``dydt`` don't need to know whether a
        parameter is stored as a constant, a callable, or a Forcing object.
        """
        if name not in self.param_values:
            raise KeyError(f"Parameter '{name}' not found in param_values.")
        return self._resolve_param(self.param_values[name], t, state)




    def get_param_vector(self, name, t, state, size):
        """Resolve a parameter and broadcast it to a fixed-length vector.

        Spatial models often allow parameters to be either a single scalar
        (applied uniformly) or a full grid-length array.  This method resolves
        the parameter via ``get_param`` and then either broadcasts a scalar or
        validates that an array has the expected size, eliminating that boilerplate
        from every ``dydt`` that works on a spatial grid.
        """
        value = self.get_param_value(name, t, state)
        if np.isscalar(value):
            return np.full(int(size), float(value), dtype=float)
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != int(size):
            raise ValueError(
                f"Parameter '{name}' resolved to size {arr.size}, expected size {int(size)}."
            )
        return arr

    def set_param_value(self, name, value):
        """Add or update a parameter and keep the attribute and ``param_values`` in sync.

        ``__setattr__`` syncs the direction ``model.alpha = v → param_values``,
        but only for names already in ``param_values``.  This method handles
        the reverse and covers inserting new parameters that weren't declared
        at initialization.
        """
        self.param_values[name] = value
        setattr(self, name, value)

    def set_function(self, name, function, bind=None):
        """Swap out a model calculation function on a single instance.

        Subclassing is the right approach when a different formulation should
        apply everywhere, but for one-off experiments (e.g. testing an
        alternative albedo scheme without writing a new class) it is useful to
        replace a single method on one instance.  The ``bind`` parameter handles
        whether the replacement expects ``self`` as its first argument.

        Parameters
        ----------
        name : str
            Name of an existing callable attribute (e.g., ``calc_k``).
        function : callable
            Replacement callable.
        bind : bool or None
            ``True``: bind as instance method (expects ``self`` as first arg).
            ``False``: assign as a plain callable.
            ``None``: infer from whether the first argument is named ``self`` or ``model``.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Function name must be a non-empty string.")
        if not callable(function):
            raise TypeError(f"Replacement for '{name}' must be callable.")
        if not hasattr(self, name):
            raise AttributeError(f"Function '{name}' does not exist on {type(self).__name__}.")
        if not callable(getattr(self, name)):
            raise TypeError(f"Attribute '{name}' exists but is not callable.")

        if bind is None:
            try:
                sig = inspect.signature(function)
                positional = [
                    p for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                bind = bool(positional) and positional[0].name.lower() in ('self', 'model')
            except (TypeError, ValueError):
                bind = False

        replacement = MethodType(function, self) if bind else function
        setattr(self, name, replacement)
        return getattr(self, name)



    def dydt(self, t, y):
        """Define the system of differential equations.

        Must be overridden by every subclass.  The solver calls this at each
        timestep with the current time ``t`` and state vector ``y``, and expects
        a list of derivatives of the same length as ``y``.  Use ``get_param``
        inside the implementation to access parameters.
        """
        pass


    def integrate(self, t_span=None, y0=None, method='RK45', dt=None,
                  output_time=None, run_name=None, kwargs=None):
        """Integrate the model over a time span and return a :class:`PBOutput`.

        Parameters
        ----------
        t_span : tuple of float
            ``(t0, tf)`` integration bounds for the solver.
        y0 : array-like
            Initial conditions.  Length must match the number of integrated
            state variables.
        method : str
            Solver to use: ``'RK45'`` (default), ``'euler'``, ``'euler_maruyama'``,
            ``'heun_maruyama'``, ``'milstein'``, ``'rk4'``, or any method
            accepted by ``scipy.integrate.solve_ivp``.

            SDE solver guidance:

            * ``'euler_maruyama'`` — strong order 0.5; baseline stochastic.
            * ``'heun_maruyama'`` — strong order 1.0 for additive noise
              (diffusion independent of state); preferred for models like
              Melcher et al. (2025).
            * ``'milstein'`` — strong order 1.0 for multiplicative noise
              (diffusion depends on state); uses a finite-difference
              approximation of ``∂g/∂y``, so no analytical Jacobian is
              required.
        dt : float, optional
            Fixed timestep for ``euler``, ``euler_maruyama``, and ``rk4``.
            Required for those methods.
        output_time : array-like, optional
            If provided, the returned ``PBOutput`` is immediately reframed onto
            this time axis (e.g. to exclude a spin-up period).
            ``output.model_time`` always retains the raw solver grid.
        run_name : str, optional
            Label stored on the output.  Defaults to ``'<method>, dt=<dt>'``.
        kwargs : dict, optional
            Additional solver options.  For ``solve_ivp`` methods these are
            forwarded directly (e.g. ``rtol``, ``atol``, ``t_eval``).  For
            ``euler_maruyama``, ``random_seed`` is extracted here.  For
            ``rk4``, ``si`` (sampling interval) is extracted here.

            .. deprecated::
                Passing ``dt`` inside ``kwargs`` is deprecated.  Use the
                explicit ``dt`` parameter instead.
        """
        kwargs = dict(kwargs) if kwargs is not None else {}

        # --- dt backward compatibility ---
        if dt is None and 'dt' in kwargs:
            warnings.warn(
                "Passing 'dt' inside kwargs is deprecated and will be removed in a "
                "future version. Use the explicit parameter: integrate(..., dt=value).",
                DeprecationWarning,
                stacklevel=2,
            )
            dt = float(kwargs.pop('dt'))
        elif dt is not None:
            kwargs.pop('dt', None)  # drop if accidentally supplied in both places

        # --- validate dt for fixed-step methods ---
        if method in ('euler', 'euler_maruyama', 'heun_maruyama', 'milstein', 'rk4'):
            if dt is None:
                raise ValueError(f"method='{method}' requires a timestep; pass dt=<value>.")
            dt = float(dt)

        # --- reset accumulators for a fresh run ---
        self.time = [0]
        self.diagnostic_variables = {var: [] for var in self.diagnostic_variables}

        # --- validate and normalise initial state ---
        y0 = self.validate_initial_state(y0)
        self.y0 = y0
        self.t_span = t_span

        # --- build initial structured array (used by step-by-step models) ---
        if self.state_variables_names:
            dtype = [(var, float) for var in self.state_variables_names]
        else:
            dtype = [type(val) for val in self.y0]
        self.dtypes = dtype
        self.state_variables = (
            np.array([tuple(self.y0)], dtype=dtype) if self.state_variables_names
            else np.array(self.y0, dtype=dtype)
        )

        # --- run solver ---
        y0_integrated = y0[:len(self.integrated_state_vars)]

        if method == 'euler':
            solution = euler_method(self.dydt, t_span, y0_integrated, dt, args=self.params)

        elif method == 'euler_maruyama':
            seed = kwargs.pop('random_seed', None)
            self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
            noise_func = getattr(self, 'sde_noise', None)
            if not callable(noise_func):
                noise_func = lambda _t, x: np.zeros_like(np.asarray(x, dtype=float))
            solution = euler_maruyama_method(
                self.dydt, t_span, y0_integrated, dt,
                noise_func=noise_func, rng=self.rng, args=self.params,
            )

        elif method == 'heun_maruyama':
            seed = kwargs.pop('random_seed', None)
            self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
            noise_func = getattr(self, 'sde_noise', None)
            if not callable(noise_func):
                noise_func = lambda _t, x: np.zeros_like(np.asarray(x, dtype=float))
            solution = heun_maruyama_method(
                self.dydt, t_span, y0_integrated, dt,
                noise_func=noise_func, rng=self.rng, args=self.params,
            )

        elif method == 'milstein':
            seed = kwargs.pop('random_seed', None)
            self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
            noise_func = getattr(self, 'sde_noise', None)
            if not callable(noise_func):
                noise_func = lambda _t, x: np.zeros_like(np.asarray(x, dtype=float))
            solution = milstein_method(
                self.dydt, t_span, y0_integrated, dt,
                noise_func=noise_func, rng=self.rng, args=self.params,
            )

        elif method == 'rk4':
            if not self.uses_post_history:
                raise ValueError(
                    "method='rk4' requires uses_post_history = True on the subclass."
                )
            si = float(kwargs.pop('si', dt))
            solution = rk4_method(self.dydt, t_span, y0_integrated, dt, si=si, args=self.params)

        else:  # scipy solve_ivp
            kwargs.setdefault('dense_output', True)
            solution = solve_ivp(
                self.dydt, t_span, y0_integrated,
                method=method, args=self.params,
                **kwargs,
            )
            solution.y = solution.y.T
            dt = 'variable'

        # --- assemble output ---
        run_name = run_name if run_name is not None else f'{method}, dt={dt}'

        if self.uses_post_history:
            history = np.asarray(solution.y, dtype=float)
            if history.ndim == 1:
                history = history.reshape(-1, 1)
            self.post_integrate(solution.t, history)
        else:
            self.state_variables = self.state_variables[1:]
            self.time = np.array(self.time)
            self.diagnostic_variables = {var: np.asarray(vals)
                                         for var, vals in self.diagnostic_variables.items()}

        output = PBOutput(
            time=self.time,
            state_variables=self.state_variables,
            state_variable_names=list(self.state_variables_names),
            diagnostic_variables=self.diagnostic_variables,
            solution=solution,
            run_name=run_name,
        )
        if output_time is not None:
            output.reframe_time_axis(output_time)
        return output


    uses_post_history = False  #: Set to True in subclasses that derive output from the full solved trajectory.



    def populate_diagnostics_from_history(self, time, history):
        """Compute diagnostic variables from the full solved trajectory.

        Called by ``post_integrate`` for models where ``uses_post_history = True``.
        The base implementation does nothing because diagnostics are model-specific;
        subclasses override this to derive any quantities that can only be computed
        once the complete trajectory is available (e.g. derived fields, fluxes).
        """
        return None

    def validate_initial_state(self, y0):
        """Validate and normalize the initial state vector.

        Exists as a method so that subclasses with non-standard initial state
        requirements (e.g. a spatially discretised model that accepts a scalar
        and broadcasts it to the grid) can override it.  The base implementation
        delegates to the utility in ``utils/solver``.
        """
        return _validate_initial_state(y0, self.integrated_state_vars, self.state_variables_names)



    def post_integrate(self, time, history):
        """Orchestrate post-solve output construction for ``uses_post_history`` models.

        Some models (e.g. spatial PDEs) cannot accumulate state during the solve
        without side effects; instead they store the full trajectory and derive
        all outputs here.  This method sequences the required steps in the correct
        order: build the structured state array, set the time axis, populate
        diagnostics, and convert diagnostic lists to arrays.  It is called
        automatically by ``integrate`` when ``uses_post_history = True``.
        """
        self.state_variables = _build_state_from_history(time, history, self.state_variables_names)
        self.time = np.asarray(time, dtype=float)
        self.populate_diagnostics_from_history(time, history)
        self.diagnostic_variables = {var: np.asarray(vals)
                                     for var, vals in self.diagnostic_variables.items()}


    # def add_noise(self, var_name, noise_ts):
    #     """Add noise to a variable in the latest output.
    #
    #     Delegates to ``self.output.add_noise``.  The clean values are saved
    #     inside the output so that ``remove_noise`` can restore them.  To
    #     generate multiple stochastic realizations from the same deterministic
    #     run, capture the return value of ``integrate()`` and call
    #     ``output.add_noise`` on each copy independently.
    #     """
    #     if self.output is None:
    #         raise RuntimeError("No output available. Call integrate() first.")
    #     self.output.add_noise(var_name, noise_ts)

    # def remove_noise(self, var_name):
    #     """Restore a variable in the latest output to its pre-noise values.
    #
    #     Delegates to ``self.output.remove_noise``.
    #     """
    #     if self.output is None:
    #         raise RuntimeError("No output available. Call integrate() first.")
    #     self.output.remove_noise(var_name)


    #
    # def to_pyleo(self, var_names=None):
    #     """Export one or more variables from the latest output as pyleoclim Series.
    #
    #     Delegates to ``self.output.to_pyleo``.  Returns a single ``Series``
    #     for one variable or a ``MultipleSeries`` for several.
    #
    #     Parameters
    #     ----------
    #     var_names : str or list of str
    #         Name(s) of state or diagnostic variable(s) to export.
    #     """
    #     if self.output is None:
    #         raise RuntimeError("No output available. Call integrate() first.")
    #     return self.output.to_pyleo(var_names)
    #
    # def reframe_time_axis(self, t_eval, update_state=True):
    #     """Resample the solution onto a target time axis.
    #
    #     Delegates to ``self.output.reframe_time_axis``, which updates
    #     ``output.time`` and ``output.state_variables`` to the resampled grid
    #     while leaving ``output.model_time`` intact.  When ``update_state=True``
    #     (the default), ``self.time`` and ``self.state_variables`` are also
    #     synced to keep backward-compatible attribute access working.
    #
    #     Parameters
    #     ----------
    #     t_eval : array-like
    #         Target time axis.
    #     update_state : bool
    #         If ``True`` (default), sync ``self.time`` and
    #         ``self.state_variables`` to the reframed values after updating
    #         the output.
    #
    #     Returns
    #     -------
    #     reframed : structured ndarray or ndarray
    #         Resampled state variables on ``t_eval``.
    #     """
    #     if self.output is None:
    #         raise RuntimeError("No output available. Call integrate() first.")
    #     reframed = self.output.reframe_time_axis(t_eval)
    #     if update_state:
    #         self.time = self.output.time
    #         self.state_variables = self.output.state_variables
    #     return reframed
