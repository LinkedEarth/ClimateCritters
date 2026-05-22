from scipy.integrate import solve_ivp
import inspect
from types import MethodType
import numpy as np

from ..utils.solver import (euler_method, euler_maruyama_method, Solution,
                            validate_initial_state as _validate_initial_state,
                            build_state_from_history as _build_state_from_history)
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

        self.t_span = None
        self.y0 = None
        self.solution = None
        self.method = None
        self.time = None
        self.kwargs = None
        self.t_eval= None
        self.run_name = None
        self.time_util = lambda t: t
        self.rng = None
        self._noise_originals = {}
        self._noisy_vars = set()

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

    def get_param(self, name, t, state):
        """Resolve a named parameter to its value at the current time and state.

        This is the standard way to access parameters inside ``dydt``.  It looks
        up the value from ``param_values`` by name and delegates resolution to
        ``_resolve_param``, so callers in ``dydt`` don't need to know whether a
        parameter is stored as a constant, a callable, or a Forcing object.
        """
        if name not in self.param_values:
            raise KeyError(f"Parameter '{name}' not found in param_values.")
        return self._resolve_param(self.param_values[name], t, state)

    def set_param(self, name, value):
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

    uses_post_history = False  #: Set to True in subclasses that derive output from the full solved trajectory.

    def build_state_from_history(self, time, history):
        """Convert raw solver output into a named structured array.

        Exists as a method (rather than a direct utility call in ``post_integrate``)
        so that subclasses can override it to apply model-specific post-processing
        — for example, clipping a variable to a physically valid range after the
        solve.  The base implementation delegates to the utility in ``utils/solver``.
        """
        return _build_state_from_history(time, history, self.state_variables_names)

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

    def get_param_vector(self, name, t, state, size):
        """Resolve a parameter and broadcast it to a fixed-length vector.

        Spatial models often allow parameters to be either a single scalar
        (applied uniformly) or a full grid-length array.  This method resolves
        the parameter via ``get_param`` and then either broadcasts a scalar or
        validates that an array has the expected size, eliminating that boilerplate
        from every ``dydt`` that works on a spatial grid.
        """
        value = self.get_param(name, t, state)
        if np.isscalar(value):
            return np.full(int(size), float(value), dtype=float)
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != int(size):
            raise ValueError(
                f"Parameter '{name}' resolved to size {arr.size}, expected size {int(size)}."
            )
        return arr

    def post_integrate(self, time, history):
        """Orchestrate post-solve output construction for ``uses_post_history`` models.

        Some models (e.g. spatial PDEs) cannot accumulate state during the solve
        without side effects; instead they store the full trajectory and derive
        all outputs here.  This method sequences the required steps in the correct
        order: build the structured state array, set the time axis, populate
        diagnostics, and convert diagnostic lists to arrays.  It is called
        automatically by ``integrate`` when ``uses_post_history = True``.
        """
        self.state_variables = self.build_state_from_history(time, history)
        self.time = np.asarray(time, dtype=float)
        self.populate_diagnostics_from_history(time, history)
        self.diagnostic_variables = {var: np.asarray(vals)
                                     for var, vals in self.diagnostic_variables.items()}

    def get_series_by_name(self, var_name):
        """Return a variable's array and its storage location ('state' or 'diagnostic').

        State variables live in a structured numpy array; diagnostic variables
        live in a dict.  This method provides a single lookup that works for
        both, returning the location string so callers (e.g. ``add_noise``) know
        where to write back a modified array.
        """
        if self.state_variables is not None and self.state_variables_names and var_name in self.state_variables_names:
            return np.asarray(self.state_variables[var_name], dtype=float), "state"
        if var_name in self.diagnostic_variables:
            return np.asarray(self.diagnostic_variables[var_name], dtype=float), "diagnostic"
        raise ValueError(f"{var_name} not found in state variables or diagnostics.")

    # noise related functions
    def add_noise(self, var_name, noise_ts):
        """Add externally provided noise to an emitted variable.

        Parameters
        ----------
        var_name : str
            Name of a state or diagnostic variable.
        noise_ts : array-like
            Noise series with the same shape as the target variable.
        """
        values, location = self.get_series_by_name(var_name)
        noise_arr = np.asarray(noise_ts, dtype=float)
        if noise_arr.shape != values.shape:
            raise ValueError(
                f"Noise shape {noise_arr.shape} does not match variable shape {values.shape} for '{var_name}'."
            )

        if var_name not in self._noise_originals:
            self._noise_originals[var_name] = values.copy()

        noisy = values + noise_arr
        if location == "state":
            self.state_variables[var_name] = noisy
        else:
            self.diagnostic_variables[var_name] = noisy
        self._noisy_vars.add(var_name)

    def remove_noise(self, var_name):
        """Restore a variable to its pre-noise state.

        Paired with ``add_noise`` to allow reversible noise experiments: the
        clean array is saved on the first ``add_noise`` call and restored here.
        """
        if var_name not in self._noise_originals:
            raise ValueError(f"No stored clean version for '{var_name}'.")
        original = self._noise_originals[var_name]
        _, location = self.get_series_by_name(var_name)
        if location == "state":
            self.state_variables[var_name] = original
        else:
            self.diagnostic_variables[var_name] = original
        self._noise_originals.pop(var_name, None)
        self._noisy_vars.discard(var_name)

    def _reset_noise_overlays(self):
        """Clear noise state at the start of each integration.

        Called by ``integrate`` before the solve so that noise added to a
        previous run's output doesn't carry over into the new one.
        """
        self._noise_originals = {}
        self._noisy_vars = set()

    def integrate(self, t_span=None, y0=None, method='RK45', kwargs=None, run_name=None):
        """Integrate the model over a time span and store the results.

        This is the main entry point for running the model.  It validates the
        initial state, selects the requested solver, runs the integration, and
        then either calls ``post_integrate`` (for ``uses_post_history`` models)
        or finalises state variables and diagnostics in-place.

        Parameters
        ----------
        t_span : tuple of float
            ``(t0, tf)`` integration bounds.
        y0 : array-like
            Initial conditions.  Length must match the number of integrated
            state variables.
        method : str
            Solver to use: ``'RK45'`` (default), ``'euler'``, ``'euler_maruyama'``,
            ``'rk4'``, or any method accepted by ``scipy.integrate.solve_ivp``.
        kwargs : dict, optional
            Solver-specific options.  ``'dt'`` is required for fixed-step methods.
        run_name : str, optional
            Label stored on the model for bookkeeping.  Defaults to a string
            describing the method and timestep.
        """

        self.t_span = t_span
        self.y0 = y0
        self.solution = None
        self.method = method
        self.time = [0]
        self._reset_noise_overlays()
        # self.t_eval = None
        self.kwargs = kwargs if kwargs is not None else {}
        if self.method in ('euler', 'euler_maruyama'):
            assert 'dt' in kwargs, "Please provide a time step for the Euler method."

        y0 = self.validate_initial_state(y0)
        self.y0 = y0

        # Define the structured array
        if len(self.state_variables_names) > 0:
            dtype = [(var, float) for i, var in enumerate(self.state_variables_names)]
            self.dtypes = dtype
        else:
            dtype = [type(val) for i, val in enumerate(self.y0)]
            self.dtypes = dtype

        if len(self.state_variables_names) > 0:
            array = np.array([tuple(self.y0)], dtype=dtype)
        else:
            array = np.array(self.y0, dtype=dtype)
        self.state_variables = array

        if kwargs is None:
            kwargs = self.kwargs
        else:
            kwargs = {**self.kwargs, **kwargs}
        if self.t_eval is not None:
            kwargs['t_eval'] = self.t_eval
        if 'method' in kwargs:
            self.method = kwargs['method']

        if self.method == 'euler':
            solution = euler_method(self.dydt, self.t_span,self.y0[:len(self.integrated_state_vars)],  kwargs['dt'],
                                    args=self.params)
        elif self.method == 'euler_maruyama':
            seed = kwargs.get('random_seed', None)
            self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

            if hasattr(self, 'sde_noise') and callable(getattr(self, 'sde_noise')):
                noise_func = self.sde_noise
            else:
                noise_func = lambda _t, x: np.zeros_like(np.asarray(x, dtype=float))

            solution = euler_maruyama_method(
                self.dydt,
                self.t_span,
                self.y0[:len(self.integrated_state_vars)],
                kwargs['dt'],
                noise_func=noise_func,
                rng=self.rng,
                args=self.params,
            )

        elif self.method == 'rk4':
            if not self.uses_post_history:
                raise ValueError(
                    "method='rk4' requires a conformant dydt with no side effects. "
                    "Set uses_post_history = True on the subclass and remove any "
                    "state-appending from dydt before using this method."
                )
            if 'dt' not in kwargs:
                raise ValueError("kwargs must include 'dt' for method='rk4'.")
            dt = float(kwargs['dt'])
            si = float(kwargs.get('si', dt))
            t0, t1 = float(t_span[0]), float(t_span[1])
            total_time = t1 - t0
            if total_time <= 0:
                raise ValueError("t_span must have t_span[1] > t_span[0].")
            if si < dt:
                dt = si
                ns = 1
            else:
                ns = int(round(si / dt))
                if abs(ns * dt - si) > 1e-10 * max(1.0, abs(si)):
                    raise ValueError("si must be an integer multiple of dt for method='rk4'.")
            nt = int(round(total_time / si))
            if abs(nt * si - total_time) > 1e-10 * max(1.0, abs(total_time)):
                raise ValueError("t_span length must be an integer multiple of si for method='rk4'.")
            y = np.asarray(y0[:len(self.integrated_state_vars)], dtype=float)
            history = np.zeros((nt + 1, y.size), dtype=float)
            times = np.zeros(nt + 1, dtype=float)
            history[0] = y
            times[0] = t0
            for step in range(nt):
                base_t = t0 + step * si
                for s in range(ns):
                    t_curr = base_t + s * dt
                    k1 = np.asarray(self.dydt(t_curr, y), dtype=float)
                    k2 = np.asarray(self.dydt(t_curr + 0.5 * dt, y + 0.5 * dt * k1), dtype=float)
                    k3 = np.asarray(self.dydt(t_curr + 0.5 * dt, y + 0.5 * dt * k2), dtype=float)
                    k4 = np.asarray(self.dydt(t_curr + dt, y + dt * k3), dtype=float)
                    y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                history[step + 1] = y
                times[step + 1] = t0 + (step + 1) * si
            solution = Solution(times, history)

        else:
            solution = solve_ivp(self.dydt, self.t_span,
                                 self.y0[:len(self.integrated_state_vars)],
                                 dense_output=kwargs['dense_output'] if 'dense_output' in kwargs else True,
                                 method=self.method,
                                 args=self.params,
                                 **kwargs)
            self.kwargs['dt'] = 'variable'
            solution.y = solution.y.T

        self.run_name = run_name if run_name is not None else f'{self.method}, dt={self.kwargs["dt"]}'
        self.solution = solution
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

    def to_pyleo(self, var_names=None):
        """Export one or more model variables as pyleoclim Series objects.

        Bridges model output to the pyleoclim ecosystem for downstream
        analysis and visualisation.  Returns a single ``Series`` for one
        variable or a ``MultipleSeries`` for several.

        Parameters
        ----------
        var_names : str or list of str
            Name(s) of state or diagnostic variable(s) to export.
        """

        from pyleoclim.core import Series, MultipleSeries

        if self.time is None:
            raise ValueError("Time axis not found. Please integrate the model first.")

        if isinstance(var_names, str):
            var_names= [var_names]


        pyleo_series = []
        for var_name in var_names:
            if var_name in self.state_variables_names:
                value = self.state_variables[var_name]
            elif var_name in self.diagnostic_variables.keys():
                value = self.diagnostic_variables[var_name]
            else:
                raise ValueError(f"{var_name} not found. Please check the state variables or diagnostics.")

            time = np.asarray(self.time)
            value = np.asarray(value)
            if len(time) != len(value):
                n = min(len(time), len(value))
                time = time[:n]
                value = value[:n]
                        
            series = Series(
                time = time,
                value = value,
                value_name = var_name,
                verbose=False,
                auto_time_params=True
                )

            pyleo_series.append(series)
        if len(pyleo_series) == 1:
            return pyleo_series[0]
        else:
            return MultipleSeries(pyleo_series)

    def reframe_time_axis(self, t_eval, update_state=True):
        """Resample the solution onto a target time axis.

        Useful for comparing model output to proxy records at specific time
        points, or for aligning two model runs on a common grid.  Uses the
        dense output from ``solve_ivp`` when available (accurate); falls back
        to linear interpolation for fixed-step solvers.

        Parameters
        ----------
        t_eval : array-like
            Target time axis.
        update_state : bool
            If ``True``, overwrite ``self.time`` and ``self.state_variables``
            with the resampled values.  If ``False``, return the resampled
            array without modifying the model.

        Returns
        -------
        reframed : structured ndarray or ndarray
            Resampled state variables on ``t_eval``.
        """

        if self.solution is None:
            raise ValueError("No solution found. Please integrate the model first.")

        t_eval = np.asarray(t_eval, dtype=float)

        # Prefer solve_ivp dense output when available
        if hasattr(self.solution, 'sol') and self.solution.sol is not None:
            y_eval = self.solution.sol(t_eval).T
        else:
            # Fallback to linear interpolation (Euler / no dense output)
            t_src = np.asarray(self.solution.t, dtype=float)
            y_src = np.asarray(self.solution.y, dtype=float)
            if y_src.ndim == 1:
                y_src = y_src.reshape(-1, 1)

            y_eval = np.column_stack([
                np.interp(t_eval, t_src, y_src[:, i])
                for i in range(y_src.shape[1])
            ])

        if self.state_variables_names:
            dtype = [(var, float) for var in self.state_variables_names]
            reframed = np.zeros(len(t_eval), dtype=dtype)
            for i, var in enumerate(self.state_variables_names):
                reframed[var] = y_eval[:, i]
        else:
            reframed = y_eval

        if update_state:
            self.time = t_eval
            self.state_variables = reframed

        return reframed
