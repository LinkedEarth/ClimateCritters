"""Numerical integrators and solver support utilities.

Provides fixed-step (RK4, Euler, Euler-Maruyama, Heun-Maruyama) integrators
that return a :class:`Solution` object, plus internal helpers used by
:class:`~paleobeasts.core.PBModel` for state validation and history
reconstruction.
"""

import numpy as np


class Solution:
    """Minimal container for the output of a fixed-step integrator.

    Parameters
    ----------
    t : ndarray, shape (n_steps,)
        Time axis.
    y : ndarray, shape (n_steps, n_vars)
        State trajectory; row ``i`` is the state at ``t[i]``.
    """

    def __init__(self, t, y):
        self.t = t
        self.y = y


def validate_monotonic_grid(values, name="grid"):
    """Validate a strictly increasing 1-D coordinate array.

    Parameters
    ----------
    values : array-like
        Coordinate values to validate.
    name : str
        Label used in error messages.  Default ``'grid'``.

    Returns
    -------
    arr : ndarray
        Validated, flattened float array.

    Raises
    ------
    ValueError
        If the array has fewer than 2 points or is not strictly increasing.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two points.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def validate_layer_thicknesses(dz, n_layers=None):
    """Validate positive layer thicknesses.

    Parameters
    ----------
    dz : array-like
        Layer thickness values.
    n_layers : int or None
        Expected number of layers.  If provided, the size of ``dz`` must
        match.  Default ``None`` (no size check).

    Returns
    -------
    dz_arr : ndarray
        Validated, flattened float array.

    Raises
    ------
    ValueError
        If the size does not match ``n_layers`` or any thickness is ≤ 0.
    """
    dz_arr = np.asarray(dz, dtype=float).reshape(-1)
    if n_layers is not None and dz_arr.size != int(n_layers):
        raise ValueError(
            f"Layer thickness size {dz_arr.size} does not match n_layers={int(n_layers)}."
        )
    if np.any(dz_arr <= 0.0):
        raise ValueError("All layer thicknesses must be > 0.")
    return dz_arr


def flux_divergence(face_fluxes, dz):
    """Compute finite-volume tendency from face fluxes and layer thicknesses.

    Parameters
    ----------
    face_fluxes : array-like, shape (n_layers + 1,)
        Fluxes at cell faces.
    dz : array-like, shape (n_layers,)
        Layer thicknesses.

    Returns
    -------
    tendency : ndarray, shape (n_layers,)
        Per-layer tendency ``-(flux_out - flux_in) / dz``.

    Raises
    ------
    ValueError
        If ``face_fluxes`` does not have length ``n_layers + 1``.
    """
    flux_arr = np.asarray(face_fluxes, dtype=float).reshape(-1)
    dz_arr = np.asarray(dz, dtype=float).reshape(-1)
    if flux_arr.size != dz_arr.size + 1:
        raise ValueError("face_fluxes must have length n_layers + 1.")
    return -(flux_arr[1:] - flux_arr[:-1]) / dz_arr


def define_t_eval(t_span, delta_t=None, num_points=None):
    """Build a ``t_eval`` array for use with :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    t_span : tuple of float
        ``(t0, tf)`` integration bounds.
    delta_t : float or None
        Output spacing.  Used when ``num_points`` is not provided.
    num_points : int or None
        Number of evenly-spaced output points.  Takes precedence over
        ``delta_t`` when both are provided.

    Returns
    -------
    t_eval : ndarray
        1-D array of output times.

    Raises
    ------
    ValueError
        If neither ``delta_t`` nor ``num_points`` is provided.
    """
    if num_points is not None:
        return np.linspace(t_span[0], t_span[1], num_points)
    if delta_t is not None:
        return np.arange(t_span[0], t_span[1], delta_t)
    raise ValueError("Either 'delta_t' or 'num_points' must be provided.")


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

    Raises
    ------
    ValueError
        If the length of ``y0`` is inconsistent with the declared state
        variable counts.
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
    history : array-like, shape (n_times, n_vars)
        Solution array.
    state_variables_names : list of str
        Names of state variables used to build a structured dtype.

    Returns
    -------
    state : structured ndarray or ndarray
        Structured array with named fields if ``state_variables_names`` is
        non-empty; otherwise the raw history array.
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


def rk4_method(f, t_span, y0, dt, si=None, args=()):
    """Fixed-step 4th-order Runge-Kutta integrator.

    Parameters
    ----------
    f : callable
        Derivative function with signature ``f(t, y, *args)``.
    t_span : tuple of float
        ``(t0, tf)`` integration bounds.
    y0 : array-like
        Initial state vector.
    dt : float
        Integration timestep.
    si : float or None
        Sampling interval — output is saved every ``si`` time units.
        Must be an integer multiple of ``dt``.  Defaults to ``dt``
        (every step is saved).
    args : tuple
        Extra positional arguments forwarded to ``f``.

    Returns
    -------
    solution : Solution
        Object with attributes ``t`` (time axis) and ``y`` (state trajectory).

    Raises
    ------
    ValueError
        If ``t_span`` is invalid, ``si`` is not an integer multiple of
        ``dt``, or ``t_span`` length is not an integer multiple of ``si``.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    dt = float(dt)
    si = float(si) if si is not None else dt

    if t1 - t0 <= 0:
        raise ValueError("t_span must satisfy t_span[1] > t_span[0].")

    if si < dt:
        dt = si
        ns = 1
    else:
        ns = int(round(si / dt))
        if abs(ns * dt - si) > 1e-10 * max(1.0, abs(si)):
            raise ValueError("si must be an integer multiple of dt.")

    nt = int(round((t1 - t0) / si))
    if abs(nt * si - (t1 - t0)) > 1e-10 * max(1.0, abs(t1 - t0)):
        raise ValueError("t_span length must be an integer multiple of si.")

    y = np.asarray(y0, dtype=float)
    history = np.zeros((nt + 1, y.size), dtype=float)
    times = np.zeros(nt + 1, dtype=float)
    history[0] = y
    times[0] = t0

    for step in range(nt):
        base_t = t0 + step * si
        for s in range(ns):
            t_curr = base_t + s * dt
            k1 = np.asarray(f(t_curr, y, *args), dtype=float)
            k2 = np.asarray(f(t_curr + 0.5 * dt, y + 0.5 * dt * k1, *args), dtype=float)
            k3 = np.asarray(f(t_curr + 0.5 * dt, y + 0.5 * dt * k2, *args), dtype=float)
            k4 = np.asarray(f(t_curr + dt, y + dt * k3, *args), dtype=float)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        history[step + 1] = y
        times[step + 1] = t0 + (step + 1) * si

    return Solution(times, history)


def euler_method(f, t_span, y0, dt, args=()):
    """Fixed-step forward Euler integrator.

    Parameters
    ----------
    f : callable
        Derivative function with signature ``f(t, y, *args)``.
    t_span : tuple of float
        ``(t0, tf)`` integration bounds.
    y0 : array-like
        Initial state vector.
    dt : float
        Fixed timestep.
    args : tuple
        Extra positional arguments forwarded to ``f``.

    Returns
    -------
    solution : Solution
        Object with attributes ``t`` (time axis) and ``y`` (state trajectory).

    Notes
    -----
    Forward Euler is first-order accurate and can be unstable for stiff
    systems or large ``dt``.  Prefer :func:`rk4_method` for most applications.
    """
    n_steps = int((t_span[1] - t_span[0]) / dt) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    for i in range(1, n_steps):
        dy = f(t[i - 1], y[i - 1], *args)
        y[i] = y[i - 1] + np.multiply(dy, dt)

    return Solution(t, y)


def euler_maruyama_method(f, t_span, y0, dt, noise_func=None, rng=None, args=()):
    """Fixed-step Euler-Maruyama integrator for stochastic differential equations.

    Solves:

        dy = f(t, y) dt + noise_func(t, y) dW

    where ``dW`` is a Wiener increment with variance ``dt``.

    Parameters
    ----------
    f : callable
        Drift function with signature ``f(t, y, *args)``.
    t_span : tuple of float
        ``(t0, tf)`` integration bounds.
    y0 : array-like
        Initial state vector.
    dt : float
        Fixed timestep.
    noise_func : callable or None
        Diffusion function with signature ``noise_func(t, y)``, returning a
        vector of per-state diffusion scales.  If ``None``, the stochastic
        term is zero and deterministic Euler is recovered.
    rng : numpy.random.Generator or None
        Random generator for Wiener increments.  A fresh generator is
        created if ``None``.
    args : tuple
        Extra positional arguments forwarded to ``f``.

    Returns
    -------
    solution : Solution
        Object with attributes ``t`` (time axis) and ``y`` (state trajectory).

    Raises
    ------
    ValueError
        If ``noise_func`` returns a vector whose shape does not match the
        state vector.
    """
    n_steps = int((t_span[1] - t_span[0]) / dt) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    if rng is None:
        rng = np.random.default_rng()

    sqrt_dt = np.sqrt(dt)

    for i in range(1, n_steps):
        t_prev, y_prev = t[i - 1], y[i - 1]
        dy = np.asarray(f(t_prev, y_prev, *args), dtype=float)

        if noise_func is None:
            diffusion = np.zeros_like(y_prev, dtype=float)
        else:
            diffusion = np.asarray(noise_func(t_prev, y_prev), dtype=float)
            if diffusion.shape != y_prev.shape:
                raise ValueError(
                    "noise_func must return a diffusion vector with the same shape as the state."
                )

        dW = rng.normal(0.0, 1.0, size=len(y_prev)) * sqrt_dt
        y[i] = y_prev + dy * dt + diffusion * dW

    return Solution(t, y)


def heun_maruyama_method(f, t_span, y0, dt, noise_func=None, rng=None, args=()):
    """Fixed-step Heun-Maruyama integrator for stochastic differential equations.

    A predictor-corrector scheme that achieves strong order 1.0 for SDEs with
    additive noise (diffusion independent of state) and weak order 2.0.  This
    is a meaningful improvement over :func:`euler_maruyama_method` (strong
    order 0.5) when transition timing is the quantity of interest, as in
    bistable climate models.

    Solves:

        dy = f(t, y) dt + g(t, y) dW

    using the two-stage Heun scheme::

        ỹ  = y  + f(t, y) dt + g(t, y) dW          (Euler predictor)
        y' = y  + ½[f(t, y) + f(t+dt, ỹ)] dt
                + ½[g(t, y) + g(t+dt, ỹ)] dW       (Heun corrector)

    A single Wiener increment ``dW`` is shared between predictor and corrector
    steps, which is the standard approach for strong convergence.

    Parameters
    ----------
    f : callable
        Drift function with signature ``f(t, y, *args)``.
    t_span : tuple of float
        ``(t0, tf)`` integration bounds.
    y0 : array-like
        Initial state vector.
    dt : float
        Fixed timestep.
    noise_func : callable or None
        Diffusion function with signature ``noise_func(t, y)``, returning a
        vector of per-state diffusion scales.  If ``None``, the stochastic
        term is zero and the Heun deterministic ODE solver is recovered.
    rng : numpy.random.Generator or None
        Random generator for Wiener increments.  A fresh generator is
        created if ``None``.
    args : tuple
        Extra positional arguments forwarded to ``f``.

    Returns
    -------
    solution : Solution
        Object with attributes ``t`` (time axis) and ``y`` (state trajectory).

    Raises
    ------
    ValueError
        If ``noise_func`` returns a vector whose shape does not match the
        state vector.

    Notes
    -----
    For purely additive noise (``g`` constant or state-independent) this
    scheme achieves strong order 1.0.  For multiplicative noise (``g``
    depends on ``y``) strong order drops back toward 0.5 but weak order 2.0
    is retained, still outperforming Euler-Maruyama in distribution-level
    statistics.  See Rößler (2010) for a full convergence analysis.
    """
    n_steps = int((t_span[1] - t_span[0]) / dt) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    if rng is None:
        rng = np.random.default_rng()

    sqrt_dt = np.sqrt(dt)

    for i in range(1, n_steps):
        t_curr, y_curr = t[i - 1], y[i - 1]
        t_next = t_curr + dt

        f0 = np.asarray(f(t_curr, y_curr, *args), dtype=float)

        if noise_func is None:
            g0 = np.zeros_like(y_curr, dtype=float)
        else:
            g0 = np.asarray(noise_func(t_curr, y_curr), dtype=float)
            if g0.shape != y_curr.shape:
                raise ValueError(
                    "noise_func must return a diffusion vector with the same shape as the state."
                )

        dW = rng.normal(0.0, 1.0, size=len(y_curr)) * sqrt_dt

        # Euler predictor
        y_pred = y_curr + f0 * dt + g0 * dW

        # Evaluate drift and diffusion at predicted state
        f1 = np.asarray(f(t_next, y_pred, *args), dtype=float)

        if noise_func is None:
            g1 = np.zeros_like(y_curr, dtype=float)
        else:
            g1 = np.asarray(noise_func(t_next, y_pred), dtype=float)

        # Heun corrector
        y[i] = y_curr + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1) * dW

    return Solution(t, y)
