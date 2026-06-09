"""Convenience factories for common :class:`~climatecritters.core.Forcing` patterns.

All public functions return either a plain callable or a
:class:`~climatecritters.core.Forcing` object.  Attach the result to a model
parameter or state variable after construction using
``model.register_forcing(var_name, forcing_obj)``.
"""

import numpy as np

from ..core.forcing import Forcing

__all__ = [
    "create_periodic_forcing_function",
    "create_periodic_forcing",
    "create_constant_forcing",
    "create_sinusoid_forcing",
    "create_piecewise_forcing",
]


def _validate_periods_powers(periods_powers):
    """Validate and normalise a sequence of (period, power) pairs."""
    if periods_powers is None:
        raise ValueError("periods_powers must not be None.")

    pairs = list(periods_powers)
    if len(pairs) == 0:
        raise ValueError("periods_powers must contain at least one (period, power) pair.")

    normalized = []
    for item in pairs:
        if len(item) != 2:
            raise ValueError("Each periods_powers entry must be a 2-tuple: (period, power).")
        period, power = float(item[0]), float(item[1])
        if period <= 0.0:
            raise ValueError("Periods must be > 0.")
        normalized.append((period, power))

    total_max_amplitude = float(sum(power for _, power in normalized))
    if np.isclose(total_max_amplitude, 0.0):
        raise ValueError("Sum of powers must be non-zero.")

    return normalized, total_max_amplitude


def create_periodic_forcing_function(periods_powers, desired_amplitude=1, y0=0):
    """Build a composite periodic forcing callable from sine components.

    Component amplitudes are rescaled so their summed peak amplitude equals
    ``desired_amplitude``.

    Parameters
    ----------
    periods_powers : sequence of (float, float)
        Sequence of ``(period, power)`` pairs.  Each ``period`` must be > 0.
        ``power`` sets the relative amplitude of that component; components
        are normalised so the total peak amplitude equals ``desired_amplitude``.
    desired_amplitude : float
        Peak amplitude of the composite signal.  Default 1.
    y0 : float
        Constant offset added to the output.  Default 0.

    Returns
    -------
    forcing_function : callable
        Function with signature ``f(t) -> float | ndarray``.  Accepts
        scalar or array ``t`` and returns the same shape.

    See also
    --------
    create_periodic_forcing : Wraps the returned callable in a
        :class:`~climatecritters.core.Forcing` object.

    Examples
    --------
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from climatecritters.utils.forcing import create_periodic_forcing_function

    # Milankovitch-like forcing: 100 kyr + 41 kyr components
    f = create_periodic_forcing_function(
        [(100, 0.6), (41, 0.4)], desired_amplitude=25.0
    )
    t = np.linspace(0, 500, 1000)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, [f(ti) for ti in t])
    ax.set_xlabel('time (kyr)'); ax.set_ylabel('forcing')
    ax.set_title('Composite periodic forcing (100 + 41 kyr)')
    plt.savefig('docs/reference/figures/create_periodic_forcing_function_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    pairs, total_max_amplitude = _validate_periods_powers(periods_powers)
    desired_amplitude = float(desired_amplitude)
    y0 = float(y0)

    def forcing_function(t):
        t_arr = np.asarray(t, dtype=float)
        result = np.full(t_arr.shape, y0, dtype=float)
        for period, power in pairs:
            frequency = 1.0 / period
            scaled_power = power / total_max_amplitude * desired_amplitude
            result += scaled_power * np.sin(2.0 * np.pi * frequency * t_arr)
        if t_arr.ndim == 0:
            return float(result)
        return result

    return forcing_function


def create_periodic_forcing(periods_powers, desired_amplitude=1, y0=0):
    """Build a composite periodic :class:`~climatecritters.core.Forcing` from sine components.

    Thin wrapper around :func:`create_periodic_forcing_function` that returns
    a :class:`~climatecritters.core.Forcing` object instead of a bare callable.

    Parameters
    ----------
    periods_powers : sequence of (float, float)
        Sequence of ``(period, power)`` pairs.  See
        :func:`create_periodic_forcing_function` for details.
    desired_amplitude : float
        Peak amplitude of the composite signal.  Default 1.
    y0 : float
        Constant offset.  Default 0.

    Returns
    -------
    forcing : cc.core.Forcing
        Forcing object wrapping the composite periodic function.

    See also
    --------
    create_periodic_forcing_function : Returns the bare callable.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    from climatecritters.utils.forcing import create_periodic_forcing
    from climatecritters.model_critters.stommel import Stommel

    orbital = create_periodic_forcing([(100, 0.6), (41, 0.4)], desired_amplitude=0.3)
    model = Stommel(E=0.0, T_star=1.0, S_star=0.0)
    model.register_forcing('E', orbital)
    output = model.integrate(t_span=(0, 500), y0=[1.0, 0.0], method='RK45')
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(output.time, output.diagnostic_variables['q'])
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('time'); ax.set_ylabel('q (overturning)')
    ax.set_title('Stommel with periodic freshwater forcing')
    plt.savefig('docs/reference/figures/create_periodic_forcing_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    func = create_periodic_forcing_function(
        periods_powers, desired_amplitude=desired_amplitude, y0=y0
    )
    return Forcing(func)


def create_constant_forcing(value):
    """Build a constant :class:`~climatecritters.core.Forcing`.

    Parameters
    ----------
    value : float
        The constant forcing value returned for all ``t``.

    Returns
    -------
    forcing : cc.core.Forcing
        Forcing object that always returns ``value``.

    Examples
    --------
    ```python
    from climatecritters.utils.forcing import create_constant_forcing

    f = create_constant_forcing(8.0)
    print(f.get_forcing(0.0))    # 8.0
    print(f.get_forcing(999.9))  # 8.0
    ```
    """
    def _constant(t):
        t_arr = np.asarray(t, dtype=float)
        out = np.full(t_arr.shape, float(value), dtype=float)
        if t_arr.ndim == 0:
            return float(out)
        return out

    return Forcing(_constant)


def create_sinusoid_forcing(A, period, y0=0.0):
    """Build a sinusoidal :class:`~climatecritters.core.Forcing`.

    Returns a forcing object representing:

        f(t) = y0 + A * sin(2 * pi * t / period)

    Parameters
    ----------
    A : float
        Amplitude of the sinusoid.
    period : float
        Period of the sinusoid (same time units as the model).  Must be > 0.
    y0 : float
        Constant offset.  Default 0.0.

    Returns
    -------
    forcing : cc.core.Forcing
        Forcing object wrapping the sinusoidal function.

    Raises
    ------
    ValueError
        If ``period`` is ≤ 0.

    Examples
    --------
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from climatecritters.utils.forcing import create_sinusoid_forcing

    seasonal = create_sinusoid_forcing(A=0.5, period=1.0)
    t = np.linspace(0, 3, 300)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, [seasonal.get_forcing(ti) for ti in t])
    ax.set_xlabel('time'); ax.set_ylabel('forcing')
    ax.set_title('Sinusoidal forcing (A=0.5, period=1)')
    plt.savefig('docs/reference/figures/create_sinusoid_forcing_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    A = float(A)
    period = float(period)
    y0 = float(y0)
    if period <= 0.0:
        raise ValueError("period must be > 0.")

    def _sinusoid(t):
        t_arr = np.asarray(t, dtype=float)
        out = y0 + A * np.sin(2.0 * np.pi * t_arr / period)
        if t_arr.ndim == 0:
            return float(out)
        return out

    return Forcing(_sinusoid)


def create_piecewise_forcing(elements, y0=0.0, label="forcing"):
    """Build a piecewise :class:`~climatecritters.core.Forcing` from a sequence of elements.

    Delegates directly to :meth:`~climatecritters.core.Forcing.from_elements`.

    Parameters
    ----------
    elements : sequence
        Ordered sequence of forcing elements (dict specs or
        :class:`~climatecritters.core.Forcing`-compatible objects) defining each
        piecewise segment.  See :meth:`~climatecritters.core.Forcing.from_elements`
        for the expected format.
    y0 : float
        Initial value before the first element takes effect.  Default 0.0.
    label : str
        Human-readable label for the forcing object.  Default ``'forcing'``.

    Returns
    -------
    forcing : cc.core.Forcing
        Piecewise forcing object.

    See also
    --------
    climatecritters.core.Forcing.from_elements : Underlying constructor.

    Examples
    --------
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import climatecritters as cc
    from climatecritters.utils.forcing import create_piecewise_forcing

    forcing = create_piecewise_forcing(
        [cc.core.Hold(duration=50,  value=0.0),
         cc.core.Ramp(duration=100, y0=0.0, yf=4.0),
         cc.core.Hold(duration=50,  value=4.0)],
        label="CO2 ramp"
    )
    t = np.linspace(0, 200, 500)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, [forcing.get_forcing(ti) for ti in t])
    ax.set_xlabel('time'); ax.set_ylabel('forcing (W m⁻²)')
    ax.set_title('Piecewise CO₂ ramp forcing')
    plt.savefig('docs/reference/figures/create_piecewise_forcing_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    return Forcing.from_elements(elements=elements, y0=y0, label=label)
