"""Miscellaneous mathematical utilities."""

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = [
    'make_derivative_func',
    'smooth_and_interpolate',
]


def make_derivative_func(method='numpy', derivative=None, data=None, time=None):
    """Return a callable that evaluates the derivative of a forcing at arbitrary times.

    Three usage modes:

    1. **Pass-through** — if ``derivative`` is already a callable, return it
       unchanged.
    2. **Numpy mode** — compute a finite-difference derivative via
       ``np.gradient``, then fit a :class:`scipy.interpolate.CubicSpline` to
       the result.  Suitable for non-uniformly spaced data.
    3. **Scipy mode** — fit a :class:`scipy.interpolate.CubicSpline` directly
       to ``data``, then return its analytical first derivative.  Slightly
       smoother than numpy mode for well-resolved data.

    Parameters
    ----------
    method : str
        Derivative method when ``data`` is provided.  One of ``'numpy'``
        (default) or ``'scipy'``.
    derivative : callable or None
        Pre-computed derivative function.  If provided and callable, it is
        returned immediately without further processing.  Default ``None``.
    data : array-like or None
        Forcing values used to compute the derivative numerically.  Required
        when ``derivative`` is ``None``.
    time : array-like or None
        Time axis corresponding to ``data``.  If ``None``, defaults to
        ``np.arange(len(data))``.

    Returns
    -------
    deriv_func : callable
        Callable with signature ``f(t) -> float`` that returns the derivative
        of the forcing at time ``t``.  In numpy and scipy modes this is a
        :class:`~scipy.interpolate.CubicSpline` (or its derivative object).

    Raises
    ------
    ValueError
        If ``derivative`` is provided but is not callable, if ``data`` is
        ``None`` when ``derivative`` is ``None``, or if ``method`` is not
        ``'numpy'`` or ``'scipy'``.

    Examples
    --------
    ```python
    import numpy as np
    import climatecritters as cc
    from climatecritters.utils.func import make_derivative_func
    from climatecritters.model_critters.g24 import Model3, calc_f

    # Build a Forcing from a data array (gives access to .data and .time)
    t_axis = np.linspace(-2000, 0, 4000)
    f_vals = np.array([calc_f(t) for t in t_axis])
    orb_forcing = cc.Forcing(data=f_vals, time=t_axis)

    # Compute a smooth derivative and attach it to Model3
    dfdt = make_derivative_func(method='scipy', data=orb_forcing.data,
                                time=orb_forcing.time)
    model = Model3()
    model.register_forcing('insolation', orb_forcing)
    model.set_param_value('dfdt', dfdt)
    output = model.integrate(t_span=(-2000, 0), y0=[0.0, 1], method='RK45',
                             kwargs={'max_step': 0.5})
    ```
    """
    if derivative is not None:
        if callable(derivative):
            return derivative
        raise ValueError(
            "'derivative' must be a callable when provided; "
            f"got {type(derivative).__name__}."
        )

    if data is None:
        raise ValueError("'data' must be provided when 'derivative' is None.")

    data = np.asarray(data, dtype=float)
    if time is None:
        time = np.arange(len(data), dtype=float)
    else:
        time = np.asarray(time, dtype=float)

    if method == 'numpy':
        numeric_derivative = np.gradient(data, time)
        return CubicSpline(time, numeric_derivative)

    if method == 'scipy':
        return CubicSpline(time, data).derivative(nu=1)

    raise ValueError(f"method must be 'numpy' or 'scipy'; got '{method}'.")


def smooth_and_interpolate(years, values, target_years=None, window=50):
    """Apply a centered moving-average and interpolate onto a target time axis.

    Parameters
    ----------
    years : array-like
        Input time axis.
    values : array-like
        Values to smooth and interpolate.
    target_years : array-like or None
        Target time axis for interpolation.  Defaults to integer annual
        spacing spanning the input range.
    window : int
        Moving-average window length.  Values ≤ 1 disable smoothing.
        Default 50.

    Returns
    -------
    smoothed_interp : ndarray
        Values smoothed and interpolated onto ``target_years``.

    Examples
    --------
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from climatecritters.utils.func import smooth_and_interpolate

    years = np.linspace(-800, 0, 1600)
    values = np.sin(2 * np.pi * years / 100) + 0.3 * np.random.default_rng(0).standard_normal(1600)
    smoothed = smooth_and_interpolate(years, values, window=30)
    target = np.arange(-800, 1, 1.0)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(years, values, lw=0.5, alpha=0.5, label='raw')
    ax.plot(target, smoothed, lw=1.5, label='smoothed (window=30)')
    ax.set_xlabel('years'); ax.legend(fontsize=9)
    ax.set_title('smooth_and_interpolate')
    plt.savefig('docs/reference/figures/smooth_and_interpolate_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)

    if target_years is None:
        target_years = np.arange(int(years.min()), int(years.max()) + 1, 1.0)
    target_years = np.asarray(target_years, dtype=float)

    if int(window) <= 1:
        smoothed = values
    else:
        kernel = np.ones(int(window), dtype=float) / float(window)
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')

    return np.interp(target_years, years, smoothed)
