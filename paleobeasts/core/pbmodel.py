from scipy.integrate import solve_ivp
import inspect
from types import MethodType
import warnings
import numpy as np
from ..utils.solver import euler_method, euler_maruyama_method, Solution
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
                 diagnostic_variables=None, parameter_contract='legacy'):
        if parameter_contract not in ('legacy', 'strict'):
            raise ValueError(
                "parameter_contract must be either 'legacy' or 'strict'."
            )
        self.variable_name = variable_name
        self.forcing = forcing
        self.parameter_contract = parameter_contract

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
        self._warned_legacy_params = set()

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
        object.__setattr__(self, name, value)
        param_values = self.__dict__.get('param_values', None)
        if isinstance(param_values, dict) and name in param_values:
            param_values[name] = value

    def resolve_param(self, param, t, state):
        """Resolve a parameter value at time t for the given state.

        Supports constants, callables, and objects with ``get_forcing``.
        """
        if param is None:
            return None
        if hasattr(param, 'get_forcing'):
            return param.get_forcing(self.time_util(t))
        if callable(param):
            if self.parameter_contract == 'strict':
                return self._call_param_strict(param, t, state)
            return self._call_param(param, t, state)
        return param

    def _strict_positional_params(self, param):
        sig = inspect.signature(param)
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
            raise TypeError(
                "Strict parameter contract does not allow var-positional (*args) callables."
            )
        return [
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

    def _is_strict_compatible_callable(self, param):
        try:
            params = self._strict_positional_params(param)
        except (TypeError, ValueError):
            return False
        n_positional = len(params)
        if n_positional not in (1, 2, 3):
            return False
        return params[0].name.lower() in ('t', 'time')

    def _call_param_strict(self, param, t, state):
        try:
            params = self._strict_positional_params(param)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Strict parameter contract requires inspectable callables with signature "
                "(t), (t, state), or (t, state, model)."
            ) from exc

        n_positional = len(params)
        if n_positional in (1, 2, 3) and params[0].name.lower() not in ('t', 'time'):
            raise TypeError(
                "Strict parameter contract requires the first argument to be time "
                "(e.g., `t` or `time`)."
            )

        if n_positional == 1:
            return param(t)
        if n_positional == 2:
            return param(t, state)
        if n_positional == 3:
            return param(t, state, self)
        raise TypeError(
            "Strict parameter contract only supports signatures "
            "(t), (t, state), or (t, state, model)."
        )

    def _warn_legacy_param_callable(self, param):
        key = id(param)
        if key in self._warned_legacy_params:
            return
        self._warned_legacy_params.add(key)
        warnings.warn(
            "Legacy parameter-callable dispatch is deprecated. "
            "Use strict signatures: (t), (t, state), or (t, state, model).",
            DeprecationWarning,
            stacklevel=3,
        )

    def _call_param(self, param, t, state):
        """Call a parameter function using a flexible signature heuristic."""
        if not self._is_strict_compatible_callable(param):
            self._warn_legacy_param_callable(param)

        try:
            sig = inspect.signature(param)
        except (TypeError, ValueError):
            return param(t, state, self)

        params = [
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if not params:
            return param()

        first_name = params[0].name.lower()

        if first_name in ('t', 'time'):
            if len(params) >= 3:
                return param(t, state, self)
            if len(params) == 2:
                return param(t, state)
            return param(t)

        if first_name in ('model', 'self', 'ebm_model'):
            if len(params) >= 2:
                return param(self, state)
            return param(self)

        if len(params) >= 2:
            second_name = params[1].name.lower()
            if second_name in ('t', 'time'):
                return param(state, t)
            if second_name in ('model', 'self', 'ebm_model'):
                return param(state, self)

        return param(state)

    def get_param(self, name, t, state):
        """Fetch a named parameter from ``param_values`` and resolve it."""
        if name not in self.param_values:
            raise KeyError(f"Parameter '{name}' not found in param_values.")
        return self.resolve_param(self.param_values[name], t, state)

    def set_param(self, name, value):
        """Set a parameter value and keep ``param_values`` in sync.

        This is a convenience for users who update parameters after model
        initialization. The solver always reads from ``param_values``, so
        this avoids subtle mismatches when mutating attributes directly.
        """
        self.param_values[name] = value
        setattr(self, name, value)

    def set_function(self, name, function, bind=None):
        """Replace a model function on this instance.

        Parameters
        ----------
        name : str
            Existing function attribute name (e.g., ``calc_k``).
        function : callable
            Replacement callable.
        bind : bool or None
            - ``True``: bind as instance method (expects ``self`` first)
            - ``False``: assign as plain callable
            - ``None``: infer from first argument name (``self``/``model`` => bind)
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

    def __copy__(self):
        new_obj = type(self)(self.forcing, self.variable_name)  # create a new instance of the same class
        new_obj.__dict__.update(self.__dict__)  # copy instance attributes

        new_obj.diagnostic_variables = {var: [] for var in self.diagnostic_variables}
        return new_obj

    def dydt(self):
        '''The differential equation of the model. 
        
        This method should be implemented and used from the child class.'''
        pass

    def uses_post_history(self):
        """Whether solved trajectories should populate state/diagnostics post-solve."""
        return False

    def build_state_from_history(self, time, history):
        """Build structured state output from a solved trajectory."""
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)

        if self.state_variables_names:
            dtype = [(var, float) for var in self.state_variables_names]
            state = np.zeros(len(time), dtype=dtype)
            for i, var in enumerate(self.state_variables_names):
                state[var] = history[:, i]
            return state
        return history

    def populate_diagnostics_from_history(self, time, history):
        """Populate diagnostic arrays from a solved trajectory."""
        return None

    def finalize_diagnostics(self):
        """Convert diagnostic containers to NumPy arrays."""
        for var in self.diagnostic_variables.keys():
            self.diagnostic_variables[var] = np.asarray(self.diagnostic_variables[var])

    def validate_initial_state(self, y0):
        """Validate and normalize the initial state vector."""
        y0_arr = np.asarray(y0, dtype=float).reshape(-1)
        n_integrated = len(self.integrated_state_vars)
        if n_integrated > 0 and y0_arr.size < n_integrated:
            raise ValueError(
                f"Initial state length {y0_arr.size} is smaller than the number of integrated "
                f"state variables ({n_integrated})."
            )
        if len(self.state_variables_names) > 0 and y0_arr.size != len(self.state_variables_names):
            raise ValueError(
                f"Initial state length {y0_arr.size} does not match declared state variable "
                f"count ({len(self.state_variables_names)})."
            )
        return y0_arr

    def get_param_vector(self, name, t, state, size):
        """Resolve a parameter and broadcast it to a fixed vector length."""
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
        """Post-solve hook for models that derive outputs from solved histories."""
        self.state_variables = self.build_state_from_history(time, history)
        self.time = np.asarray(time, dtype=float)
        self.populate_diagnostics_from_history(time, history)
        self.finalize_diagnostics()

    def get_series_by_name(self, var_name):
        if self.state_variables is not None and self.state_variables_names and var_name in self.state_variables_names:
            return np.asarray(self.state_variables[var_name], dtype=float), "state"
        if var_name in self.diagnostic_variables:
            return np.asarray(self.diagnostic_variables[var_name], dtype=float), "diagnostic"
        raise ValueError(f"{var_name} not found in state variables or diagnostics.")

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
        """Restore a variable previously modified with ``add_noise``."""
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
        self._noise_originals = {}
        self._noisy_vars = set()

    def integrate(self, t_span=None, y0=None, method='RK45', kwargs=None, run_name=None):
        '''Integrates the model over a given time span.
        
        Parameters
        ----------
        
        t_span : tuple, list
            The time span over which the model will be integrated.
            
        y0 : list
            Initial conditions for the model. The length of this list should be equal to the number of model state variables.
        
        method : str
            The integration method to be use; options include 'RK45' (Runge Kutta),
            'euler', and 'euler_maruyama'. Default is 'RK45'.

        kwargs : dict
            Additional keyword arguments to be passed to the solver.


        '''

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
            if not self.uses_post_history():
                raise ValueError(
                    "method='rk4' requires a conformant dydt with no side effects. "
                    "Override uses_post_history() to return True and remove any "
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
        if self.uses_post_history():
            history = np.asarray(solution.y, dtype=float)
            if history.ndim == 1:
                history = history.reshape(-1, 1)
            self.post_integrate(solution.t, history)
        else:
            self.state_variables = self.state_variables[1:]
            self.time = np.array(self.time)

            self.finalize_diagnostics()

    def to_pyleo(self,var_names=None):
        '''Function to create a pyleoclim Series object from a state variable.

        Parameters
        ----------

        var_names : str or list
            The name(s) of the state or diagnostic variable(s) to be converted.'''

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
        '''Reframe the solution onto a specified time axis.

        Parameters
        ----------
        t_eval : array-like
            Target time axis for resampling.

        update_state : bool
            If True, update self.time and self.state_variables to the reframed values.
            If False, leave the model state intact and return the reframed values only.

        Returns
        -------
        reframed : structured array or ndarray
            The reframed state variables on t_eval. Structured array if the model
            has named state variables, otherwise a 2D ndarray.
        '''

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
