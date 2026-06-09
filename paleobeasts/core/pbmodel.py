from scipy.integrate import solve_ivp
import inspect
import re
import textwrap
import warnings
from types import MethodType
import numpy as np

from ..utils.solver import (euler_method, euler_maruyama_method, heun_maruyama_method,
                            milstein_method, rk4_method, Solution,
                            validate_initial_state as _validate_initial_state,
                            build_state_from_history as _build_state_from_history)
from .pboutput import PBOutput
from .forcing import ForcingSpec


def _parse_docstring_params(cls):
    """Collect {name: description} from NumPy-style Parameters sections in the MRO.

    Walks the MRO from base to derived so that subclass entries override base
    class entries for the same parameter name.
    """
    result = {}
    for klass in reversed(cls.__mro__):
        raw = inspect.getdoc(klass)
        if not raw:
            continue
        lines = raw.splitlines()
        i = 0
        while i < len(lines):
            # NumPy section header: word(s) followed by a line of dashes
            if (lines[i].strip() == 'Parameters'
                    and i + 1 < len(lines)
                    and re.match(r'^-{3,}\s*$', lines[i + 1])):
                i += 2  # skip "Parameters" + dashes
                while i < len(lines):
                    line = lines[i]
                    # Next section starts: non-indented text followed by dashes
                    if (line.strip()
                            and not line[0].isspace()
                            and i + 1 < len(lines)
                            and re.match(r'^-{3,}\s*$', lines[i + 1])):
                        break
                    # Parameter entry: "name : ..." not indented
                    m = re.match(r'^(\w+)\s*:', line)
                    if m and not line[0].isspace():
                        name = m.group(1)
                        i += 1
                        desc_parts = []
                        while i < len(lines) and (
                                not lines[i].strip() or lines[i][0].isspace()):
                            stripped = lines[i].strip()
                            if stripped:
                                desc_parts.append(stripped)
                            i += 1
                        result[name] = ' '.join(desc_parts)
                    else:
                        i += 1
                break
            i += 1
    return result


def _format_value(v):
    """Return a compact display string for a parameter value."""
    if callable(v) and not hasattr(v, 'get_forcing'):
        name = getattr(v, '__name__', None) or getattr(type(v), '__name__', None)
        return f'<callable: {name}>' if name else '<callable>'
    if hasattr(v, 'get_forcing'):
        return f'<{type(v).__name__}>'
    if isinstance(v, float):
        return f'{v:g}'
    return repr(v)


class PBModel:
    """The overarching model structure for PaleoBeasts.

    PBModel serves as the archetype/parent class for models within the
    ``signal_models`` directory.  It is not meant to be instantiated directly.

    Parameter handling
    ------------------
    Models may define a ``param_values`` dict that maps parameter names to
    constants, callables, or ``Forcing`` objects.  Use
    ``get_param_value(name, t, state)`` inside ``dydt`` to resolve any of these
    uniformly.

    To make a parameter state- or time-dependent after construction, assign a
    callable via ``set_param_value``::

        model.set_param_value('rho', lambda t, state: 28.0 + 0.1 * state[0])

    To replace a *computation method* (e.g. ``calc_albedo``) on a single
    instance without subclassing, use ``set_function``.

    External forcings
    -----------------
    Use ``register_forcing`` to attach a time-varying external driver to any
    named parameter or state variable after construction::

        model.register_forcing('S0', forcing_obj)                         # parameter
        model.register_forcing('x', forcing_obj, 'additive', timing='pre')  # state

    See ``register_forcing`` for the full contract and default timing rules.
    """

    def __init__(self, variable_name, state_variables=None, non_integrated_state_vars=None,
                 diagnostic_variables=None):
        self.variable_name = variable_name
        self._forcings: dict = {}

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

        Uses ``object.__new__`` rather than calling the constructor so that
        subclass ``__init__`` signatures don't need to be reproduced here.
        ``_forcings`` lists are copied shallowly — the ``ForcingSpec`` objects
        themselves are shared (they carry no mutable state).
        """
        new_obj = object.__new__(type(self))
        new_obj.__dict__.update(self.__dict__)
        new_obj.diagnostic_variables = {var: [] for var in self.diagnostic_variables}
        new_obj._forcings = {k: list(v) for k, v in self._forcings.items()}
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

    def register_forcing(self, var_name: str, forcing_object, attachment_style: str = None,
                         timing: str = None):
        """Attach an external forcing to a named parameter or state variable.

        Parameters
        ----------
        var_name : str
            Name of the target.  Must exist in ``param_values`` (parameter
            namespace) or ``state_variables_names`` (state namespace).  If the
            name appears in both, a ``ValueError`` is raised — this is a model
            design issue worth resolving explicitly.
        forcing_object :
            A ``Forcing`` instance, a callable ``f(t)`` → scalar/array, or any
            object with a ``get_forcing(t)`` method.
        attachment_style : {"replacement", "additive"}, optional
            How the forcing value is applied.

            * Parameters default to ``"replacement"``.  ``"additive"`` is
              also supported and adds the forcing value to the nominal
              parameter at each step (e.g. ``k = k_0 + ε(t)``).
            * State variables have **no default** — ``attachment_style`` is
              required.  This is intentional: injecting into a live state
              variable is a significant physical choice that should be explicit.
        timing : {"pre", "post"}, optional
            When the forcing is applied relative to the integration step.
            Derived automatically in most cases:

            * parameter + any style   → ``"pre"`` (always; no override)
            * state + replacement     → ``"post"`` (always; warns if ``"pre"`` passed)
            * state + additive        → **required**; raise if not provided

        Raises
        ------
        ValueError
            If ``var_name`` is not found, if ``attachment_style`` is missing for
            a state variable, if ``timing`` is missing for state + additive, or
            if a second ``"replacement"`` is registered on the same variable.
        """
        in_params = var_name in self.param_values
        in_state = var_name in self.state_variables_names

        if in_params and in_state:
            raise ValueError(
                f"'{var_name}' appears in both param_values and state_variables_names. "
                "Resolve this ambiguity in the model definition before registering a forcing."
            )
        if not in_params and not in_state:
            valid = sorted(list(self.param_values.keys()) + list(self.state_variables_names))
            raise ValueError(
                f"'{var_name}' not found in this model's parameters or state variables. "
                f"Valid names: {valid}"
            )

        if in_params:
            if attachment_style is None:
                attachment_style = "replacement"
            if attachment_style not in ("replacement", "additive"):
                raise ValueError(
                    f"Parameters support attachment_style='replacement' or 'additive'; "
                    f"got {attachment_style!r} for '{var_name}'."
                )
            resolved_timing = "pre"
            if timing is not None and timing != "pre":
                raise ValueError(
                    f"Parameter forcings are always applied pre-step; "
                    f"timing='{timing}' is not valid for '{var_name}'."
                )

        else:  # state variable
            if attachment_style is None:
                raise ValueError(
                    f"attachment_style is required when forcing a state variable ('{var_name}'). "
                    "Choose 'replacement' (post-step correction to x) or "
                    "'additive' (pre- or post-step injection — specify timing)."
                )
            if attachment_style == "replacement":
                if timing is not None and timing != "post":
                    warnings.warn(
                        f"State variable replacement is always applied post-step. "
                        f"Ignoring timing='{timing}' for '{var_name}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                resolved_timing = "post"
            elif attachment_style == "additive":
                if timing is None:
                    raise ValueError(
                        f"timing is required for additive state forcing on '{var_name}'. "
                        "Choose 'pre' (adds to dx/dt inside the RHS) or "
                        "'post' (adds to x after each integration step)."
                    )
                resolved_timing = timing
            else:
                raise ValueError(
                    f"attachment_style must be 'replacement' or 'additive'; "
                    f"got {attachment_style!r}."
                )

        # conflict check: two replacements on the same variable
        existing = self._forcings.get(var_name, [])
        if attachment_style == "replacement" and any(
            s.attachment_style == "replacement" for s in existing
        ):
            raise ValueError(
                f"A replacement forcing is already registered for '{var_name}'. "
                "Remove it before registering another, or use attachment_style='additive'."
            )

        spec = ForcingSpec(
            forcing_object=forcing_object,
            attachment_style=attachment_style,
            timing=resolved_timing,
        )
        self._forcings.setdefault(var_name, []).append(spec)

    def get_forcings(self, var_name: str = None):
        """Return registered forcings, optionally filtered by variable name.

        Parameters
        ----------
        var_name : str, optional
            If given, return the list of ``ForcingSpec`` objects for that
            variable.  If omitted, return the full ``_forcings`` dict.
        """
        if var_name is None:
            return dict(self._forcings)
        return list(self._forcings.get(var_name, []))

    def clear_forcings(self, var_name: str = None):
        """Remove registered forcings.

        Parameters
        ----------
        var_name : str, optional
            If given, clear only the forcings for that variable.
            If omitted, clear all forcings on the model.
        """
        if var_name is None:
            self._forcings.clear()
        else:
            self._forcings.pop(var_name, None)

    # ------------------------------------------------------------------
    # Forcing application helpers
    # ------------------------------------------------------------------

    _FIXED_STEP_METHODS = {"euler", "euler_maruyama", "rk4"}

    def _build_forced_dydt(self):
        """Return a wrapped dydt that applies all pre-step forcings.

        For parameter replacement forcings: temporarily patches ``param_values``
        before calling the original ``dydt``, then restores it.  This means
        ``get_param_value`` inside ``dydt`` transparently sees the forced value
        without any changes to subclass code.

        For state additive forcings: adds the forcing value to the appropriate
        index of the returned dxdt vector after the original ``dydt`` returns.

        Returns the original ``dydt`` unchanged if no pre-step forcings are
        registered, so the caller can always substitute the result.
        """
        pre_param_replace = []   # [(var_name, spec), ...]
        pre_param_additive = []  # [(var_name, spec), ...]
        pre_state = []           # [(idx, spec), ...]

        for var_name, specs in self._forcings.items():
            for spec in specs:
                if spec.timing != "pre":
                    continue
                if var_name in self.param_values:
                    if spec.attachment_style == "additive":
                        pre_param_additive.append((var_name, spec))
                    else:
                        pre_param_replace.append((var_name, spec))
                elif var_name in self.integrated_state_vars:
                    idx = self.integrated_state_vars.index(var_name)
                    pre_state.append((idx, spec))

        if not pre_param_replace and not pre_param_additive and not pre_state:
            return self.dydt

        original_dydt = self.dydt

        def forced_dydt(t, x, *args):
            saved = {}
            t_eval = self.time_util(t)

            for var_name, spec in pre_param_replace:
                saved[var_name] = self.param_values[var_name]
                self.param_values[var_name] = spec.evaluate(t_eval)

            for var_name, spec in pre_param_additive:
                if var_name not in saved:
                    saved[var_name] = self.param_values[var_name]
                self.param_values[var_name] = self.param_values[var_name] + spec.evaluate(t_eval)

            dxdt = np.asarray(original_dydt(t, x, *args), dtype=float).copy()

            for var_name, original_val in saved.items():
                self.param_values[var_name] = original_val

            for idx, spec in pre_state:
                dxdt[idx] += spec.evaluate(t_eval)

            return dxdt

        return forced_dydt

    def _build_post_step(self):
        """Return a post-step callback for all post-step forcings, or None.

        The callback has signature ``post_step(t, y) -> y`` and is intended
        for the fixed-step solvers.  It applies replacement and additive
        forcings in registration order for each variable.

        Returns ``None`` if no post-step forcings are registered.
        """
        post_forcings = []  # [(idx, spec), ...]

        for var_name, specs in self._forcings.items():
            if var_name not in self.integrated_state_vars:
                continue
            idx = self.integrated_state_vars.index(var_name)
            for spec in specs:
                if spec.timing == "post":
                    post_forcings.append((idx, spec))

        if not post_forcings:
            return None

        def post_step(t, y):
            y = np.asarray(y, dtype=float).copy()
            for idx, spec in post_forcings:
                val = spec.evaluate(self.time_util(t))
                if spec.attachment_style == "replacement":
                    y[idx] = val
                else:
                    y[idx] += val
            return y

        return post_step

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
        
        if method == 'RK45' and kwargs is None: 
            kwargs ={'rtol': 1e-8, 'atol': 1e-10}
        else:
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
        # Use the actual start time, not a hardcoded 0, so that models with
        # t_span starting at negative times (e.g. palaeoclimate BP conventions)
        # accumulate correctly.
        self.time = [float(t_span[0])]
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

        # --- build forcing wrappers ---
        dydt_fn = self._build_forced_dydt()
        post_step = self._build_post_step()

        if post_step is not None and method not in self._FIXED_STEP_METHODS:
            warnings.warn(
                f"Post-step forcings are registered but method='{method}' is adaptive. "
                "Post-step forcings will not be applied during integration. "
                "Use a fixed-step method ('euler', 'rk4', 'euler_maruyama') to apply them.",
                UserWarning,
                stacklevel=2,
            )
            post_step = None

        # --- run solver ---
        y0_integrated = y0[:len(self.integrated_state_vars)]

        if method == 'euler':
            solution = euler_method(dydt_fn, t_span, y0_integrated, dt,
                                    args=self.params, post_step=post_step)

        elif method == 'euler_maruyama':
            seed = kwargs.pop('random_seed', None)
            self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
            noise_func = getattr(self, 'sde_noise', None)
            if not callable(noise_func):
                noise_func = lambda _t, x: np.zeros_like(np.asarray(x, dtype=float))
            solution = euler_maruyama_method(
                dydt_fn, t_span, y0_integrated, dt,
                noise_func=noise_func, rng=self.rng, args=self.params,
                post_step=post_step,
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
            solution = rk4_method(dydt_fn, t_span, y0_integrated, dt, si=si,
                                  args=self.params, post_step=post_step)

        else:  # scipy solve_ivp
            kwargs.setdefault('dense_output', True)
            solution = solve_ivp(
                dydt_fn, t_span, y0_integrated,
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
            # Sort and deduplicate by time.  Adaptive solvers (e.g. RK45) call
            # dydt at every stage evaluation including rejected steps that are
            # retried with a smaller dt.  Those ghost evaluations leave duplicate
            # or out-of-order timestamps in any arrays accumulated inside dydt.
            # np.unique returns the *first* occurrence of each unique time value,
            # which preserves the forward-integration entry over any retry.
            if len(self.state_variables) > 0:
                _, _unique_idx = np.unique(self.time, return_index=True)
                self.time = self.time[_unique_idx]
                self.state_variables = self.state_variables[_unique_idx]
                self.diagnostic_variables = {
                    k: v[_unique_idx] for k, v in self.diagnostic_variables.items()
                }
            # For scipy adaptive solvers, further restrict to the accepted step
            # endpoints stored in solution.t.  Adaptive methods (RK45, DOP853,
            # etc.) evaluate dydt at multiple intermediate stage points per step;
            # those intermediate values can stray far outside the physical range
            # before the solver corrects itself, producing spurious transients in
            # any arrays accumulated step-by-step inside dydt.
            # Fixed-step methods (euler, rk4, etc.) call dydt exactly once per
            # step so no further filtering is needed for those.
            if dt == 'variable':
                accepted_t = np.round(np.asarray(solution.t, dtype=float), 10)
                keep = np.isin(np.round(self.time, 10), accepted_t)
                if keep.sum() > 0:
                    self.time = self.time[keep]
                    self.state_variables = self.state_variables[keep]
                    self.diagnostic_variables = {
                        k: v[keep] for k, v in self.diagnostic_variables.items()
                    }

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


    # ------------------------------------------------------------------
    # Discovery / documentation helpers
    # ------------------------------------------------------------------

    _VALID_LIST_TARGETS = ('state_variables', 'parameters', 'diagnostic_variables')

    def list(self, list_target):
        """Return a list of names for the requested category.

        Parameters
        ----------
        list_target : {'state_variables', 'parameters', 'diagnostic_variables'}
            The category to enumerate.

        Returns
        -------
        names : list of str
        """
        if list_target not in self._VALID_LIST_TARGETS:
            raise ValueError(
                f"list_target must be one of {self._VALID_LIST_TARGETS!r}; "
                f"got {list_target!r}."
            )
        if list_target == 'state_variables':
            return list(self.state_variables_names)
        if list_target == 'parameters':
            return list(self.param_values.keys())
        return list(self.diagnostic_variables.keys())

    def doc(self, print_target):
        """Pretty-print documentation for the requested category.

        Descriptions are parsed from the NumPy-style docstrings of the class
        and its base classes.  For parameters the current value is also shown.

        Parameters
        ----------
        print_target : {'state_variables', 'parameters', 'diagnostic_variables'}
            The category to document.
        """
        if print_target not in self._VALID_LIST_TARGETS:
            raise ValueError(
                f"print_target must be one of {self._VALID_LIST_TARGETS!r}; "
                f"got {print_target!r}."
            )

        class_name = type(self).__name__
        descriptions = _parse_docstring_params(type(self))

        if print_target == 'parameters':
            names = list(self.param_values.keys())
            header = f'Parameters — {class_name}'
            columns = ('Name', 'Current value', 'Description')
            rows = []
            for name in names:
                val_str = _format_value(self.param_values[name])
                desc = descriptions.get(name, '')
                rows.append((name, val_str, desc))

        elif print_target == 'state_variables':
            names = list(self.state_variables_names)
            header = f'State variables — {class_name}'
            columns = ('Name', 'Kind', 'Description')
            rows = []
            for name in names:
                kind = ('non-integrated' if name in self.non_integrated_state_vars
                        else 'integrated')
                desc = descriptions.get(name, '')
                rows.append((name, kind, desc))

        else:  # diagnostic_variables
            names = list(self.diagnostic_variables.keys())
            header = f'Diagnostic variables — {class_name}'
            columns = ('Name', 'Description')
            rows = [(name, descriptions.get(name, '')) for name in names]

        self._print_table(header, columns, rows)

    @staticmethod
    def _print_table(header, columns, rows):
        """Render a fixed-width text table to stdout."""
        desc_col = len(columns) - 1   # last column always wraps
        max_desc_width = 52

        # Compute column widths (all but the last description column)
        col_widths = [len(col) for col in columns]
        for row in rows:
            for i, cell in enumerate(row[:-1]):
                col_widths[i] = max(col_widths[i], len(cell))

        # Build a format template for fixed columns
        fixed_fmt = '  '.join(f'{{:<{w}}}' for w in col_widths[:-1])
        total_fixed = sum(col_widths[:-1]) + 2 * (len(col_widths) - 2)

        desc_header_pad = max(len(columns[-1]), 0)
        total_width = total_fixed + 2 + max(desc_header_pad, max_desc_width)

        sep = '─' * total_width
        thick = '═' * total_width

        print()
        print(header)
        print(thick)

        # Header row
        if len(columns) > 1:
            header_fixed = fixed_fmt.format(*[c for c in columns[:-1]])
            print(f'{header_fixed}  {columns[-1]}')
        else:
            print(columns[0])
        print(sep)

        for row in rows:
            fixed_cells = row[:-1]
            desc = row[-1]

            wrapped = textwrap.wrap(desc, width=max_desc_width) if desc else ['']
            if len(columns) > 1:
                fixed_str = fixed_fmt.format(*fixed_cells)
                print(f'{fixed_str}  {wrapped[0]}')
                indent = ' ' * (total_fixed + 2)
                for extra in wrapped[1:]:
                    print(f'{indent}{extra}')
            else:
                print(f'{fixed_cells[0]}  {wrapped[0]}')
                for extra in wrapped[1:]:
                    print(f'{"":>{len(fixed_cells[0])+2}}{extra}')

        print(thick)
        print()

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
