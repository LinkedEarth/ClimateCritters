from scipy.integrate import solve_ivp
import numpy as np


# class Solver:
#     def __init__(self, model, t_span, y0, method='RK45', kwargs=None):
#         self.model = model
#         self.t_span = t_span
#         self.y0 = y0
#         self.solution = None
#         self.method = method
#         self.t_eval = None
#         self.kwargs = kwargs if kwargs is not None else {}
#         self.diagnostics = {}
#
#         # Define the structured array
#         if len(self.model.state_variables_names) > 0:
#             dtype = [(var, float) for i, var in enumerate(self.model.state_variables_names)]
#             self.model.dtypes = dtype
#         else:
#             dtype = [type(val) for i, val in enumerate(self.y0)]
#             self.model.dtypes = dtype
#
#         array = np.array(self.y0, dtype=dtype)
#         self.model.state_variables = array

    # def define_t_eval(self, delta_t=None, num_points=None):
    #     if num_points is not None:
    #         self.t_eval = np.linspace(self.t_span[0], self.t_span[1], num_points)
    #     elif delta_t is not None:
    #         self.t_eval = np.arange(self.t_span[0], self.t_span[1], delta_t)
    #     else:
    #         raise ValueError("Either 'delta_t' or 'num_points' must be provided.")

    # def integrate(self, kwargs=None):
    #     if kwargs is None:
    #         kwargs = self.kwargs
    #     else:
    #         kwargs = {**self.kwargs, **kwargs}
    #     if self.t_eval is not None:
    #         kwargs['t_eval'] = self.t_eval
    #     if 'method' in kwargs:
    #         self.method = kwargs['method']
    #
    #     if self.method == 'euler':
    #         solution = euler_method(self.model.dydt, self.y0[:len(self.model.integrated_state_vars)], self.t_span[0], self.t_span[1], kwargs['dt'],
    #                                 args=self.model.params)
    #
    #     else:
    #         solution = solve_ivp(self.model.dydt, self.t_span,
    #                              self.y0[:len(self.model.integrated_state_vars)],
    #                              dense_output=kwargs['dense_output'] if 'dense_output' in kwargs else True,
    #                              method=self.method,
    #                              args=self.model.params,
    #                              **kwargs)
    #         solution.y = solution.y.T
    #
    #     self.solution = solution
    #     self.model.state_variables = self.model.state_variables[1:]
    #
    #     for var in self.model.diagnostic_variables.keys():
    #         self.model.diagnostic_variables[var] = np.array(
    #             self.model.diagnostic_variables[var])  # .reshape(len(solution.y)


class Solution:
    def __init__(self, t, y):
        self.t = t
        self.y = y


def validate_monotonic_grid(values, name="grid"):
    """Validate a strictly increasing 1D coordinate array."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two points.")
    diffs = np.diff(arr)
    if np.any(diffs <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def validate_layer_thicknesses(dz, n_layers=None):
    """Validate positive layer thicknesses."""
    dz_arr = np.asarray(dz, dtype=float).reshape(-1)
    if n_layers is not None and dz_arr.size != int(n_layers):
        raise ValueError(f"Layer thickness size {dz_arr.size} does not match n_layers={int(n_layers)}.")
    if np.any(dz_arr <= 0.0):
        raise ValueError("All layer thicknesses must be > 0.")
    return dz_arr


def flux_divergence(face_fluxes, dz):
    """Compute finite-volume tendency from face fluxes and layer thicknesses."""
    flux_arr = np.asarray(face_fluxes, dtype=float).reshape(-1)
    dz_arr = np.asarray(dz, dtype=float).reshape(-1)
    if flux_arr.size != dz_arr.size + 1:
        raise ValueError("face_fluxes must have length n_layers + 1.")
    return -(flux_arr[1:] - flux_arr[:-1]) / dz_arr

def define_t_eval(t_span, delta_t=None, num_points=None):
    t_eval = None
    if num_points is not None:
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
    elif delta_t is not None:
        t_eval = np.arange(t_span[0], t_span[1], delta_t)
    else:
        raise ValueError("Either 'delta_t' or 'num_points' must be provided. Function will return None")
    return t_eval

def validate_initial_state(y0, integrated_state_vars, state_variables_names):
    """Validate and normalize the initial state vector.

    Parameters
    ----------
    y0 : array-like
        Proposed initial state.
    integrated_state_vars : list
        Names of integrated (ODE) state variables.
    state_variables_names : list
        All declared state variable names.

    Returns
    -------
    y0_arr : ndarray
        Validated, flattened float array.
    """
    y0_arr = np.asarray(y0, dtype=float).reshape(-1)
    n_integrated = len(integrated_state_vars)
    if n_integrated > 0 and y0_arr.size < n_integrated:
        raise ValueError(
            f"Initial state length {y0_arr.size} is smaller than the number of integrated "
            f"state variables ({n_integrated})."
        )
    if len(state_variables_names) > 0 and y0_arr.size != len(state_variables_names):
        raise ValueError(
            f"Initial state length {y0_arr.size} does not match declared state variable "
            f"count ({len(state_variables_names)})."
        )
    return y0_arr


def build_state_from_history(time, history, state_variables_names):
    """Build a structured state array from a solved trajectory.

    Parameters
    ----------
    time : array-like
        Time axis of the solution.
    history : array-like
        Solution array of shape (n_times, n_vars).
    state_variables_names : list of str
        Names of state variables, used to build a structured dtype.

    Returns
    -------
    state : structured ndarray or ndarray
        Structured array with named fields if state_variables_names is non-empty,
        otherwise the raw history array.
    """
    time = np.asarray(time, dtype=float)
    history = np.asarray(history, dtype=float)
    if state_variables_names:
        dtype = [(var, float) for var in state_variables_names]
        state = np.zeros(len(time), dtype=dtype)
        for i, var in enumerate(state_variables_names):
            state[var] = history[:, i]
        return state
    return history


def euler_method(f, t_span, y0, dt, args=()):
    """
    Solves an ODE using the Euler method with a fixed timestep.

    :param args:
    :param f: callable - The derivative function of the ODE (dy/dt).
    :param y0: float - Initial condition.
    :param t0: float - Initial time.
    :param tf: float - Final time.
    :param dt: float - Timestep for the integration.
    :return: Solution - An object containing the time and value arrays.
    """

    n_steps = int((t_span[1] - t_span[0]) / dt) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    for i in range(1, n_steps):
        if args is not None:
            dy = f(t[i - 1], y[i - 1], *args)
        else:
            dy = f(t[i - 1], y[i - 1])
        y[i] = y[i - 1] + np.multiply(dy, dt)

    solution = Solution(t, y)
    return solution


def euler_maruyama_method(f, t_span, y0, dt, noise_func=None, rng=None, args=()):
    """
    Solves an SDE using fixed-step Euler-Maruyama integration.

    Parameters
    ----------
    f : callable
        Drift function with signature f(t, y, *args).
    t_span : tuple[float, float]
        Integration bounds (t0, tf).
    y0 : array-like
        Initial state vector.
    dt : float
        Fixed timestep.
    noise_func : callable or None
        Function returning per-state diffusion scale at (t, y) with signature
        noise_func(t, y). If None, deterministic Euler is recovered.
    rng : np.random.Generator or None
        Random generator used for Wiener increments.
    args : tuple
        Extra positional args passed to f.
    """
    n_steps = int((t_span[1] - t_span[0]) / dt) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    if rng is None:
        rng = np.random.default_rng()

    sqrt_dt = np.sqrt(dt)

    for i in range(1, n_steps):
        t_prev = t[i - 1]
        y_prev = y[i - 1]
        if args is not None:
            dy = np.asarray(f(t_prev, y_prev, *args), dtype=float)
        else:
            dy = np.asarray(f(t_prev, y_prev), dtype=float)

        if noise_func is None:
            diffusion = np.zeros_like(y_prev, dtype=float)
        else:
            diffusion = np.asarray(noise_func(t_prev, y_prev), dtype=float)
            if diffusion.shape != y_prev.shape:
                raise ValueError(
                    "noise_func must return diffusion vector with same shape as state."
                )

        dW = rng.normal(0.0, 1.0, size=len(y_prev)) * sqrt_dt
        y[i] = y_prev + dy * dt + diffusion * dW

    solution = Solution(t, y)
    return solution
