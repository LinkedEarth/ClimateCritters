"""Convenience factories for common :class:`~climatecritters.core.Forcing` patterns.

All public functions return either a :class:`~climatecritters.core.Forcing`
(when no ``duration`` is given) or a :class:`~climatecritters.core.ForcingElement`
(when ``duration`` is given).  The unified entry point is :func:`create_forcing`;
the named factories are ergonomic aliases that build their callable internally.

.. rubric:: Duration gate

Every factory accepts an optional ``duration`` keyword:

* **No duration** → returns an indefinite :class:`~climatecritters.core.Forcing`
  backed by a lambda.  Suitable for perpetual signals (orbital forcing, seasonal
  cycle, noise) registered directly with a model.
* **With duration** → returns a bounded :class:`~climatecritters.core.ForcingElement`
  that can be composed into a :class:`~climatecritters.core.ForcingSequence` and
  then compiled::

      elem = create_sinusoid_forcing(A=5.0, period=1.0, duration=10.0)
      seq  = Hold(5, value=0.0) + elem + Hold(5, value=0.0)
      f    = seq.compile()   # → Forcing, ready to register
"""

import numpy as np

from ..core.forcing import Forcing, ForcingElement, ForcingSequence

__all__ = [
    "create_forcing",
    "create_sinusoid_forcing",
    "create_periodic_forcing",
    "create_constant_forcing",
    "create_piecewise_forcing",
    "make_forcing_element",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    total = float(sum(p for _, p in normalized))
    if np.isclose(total, 0.0):
        raise ValueError("Sum of powers must be non-zero.")
    return normalized, total


def _build_periodic_func(periods_powers, desired_amplitude=1.0, y0=0.0):
    """Return a callable ``f(t)`` for a normalised multi-frequency sine sum."""
    pairs, total = _validate_periods_powers(periods_powers)
    desired_amplitude = float(desired_amplitude)
    y0 = float(y0)

    def _func(t):
        t_arr = np.asarray(t, dtype=float)
        result = np.full(t_arr.shape, y0, dtype=float)
        for period, power in pairs:
            scaled = power / total * desired_amplitude
            result += scaled * np.sin(2.0 * np.pi * t_arr / period)
        return float(result) if t_arr.ndim == 0 else result

    return _func


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def create_forcing(func, duration=None):
    """Create a forcing from an arbitrary callable.

    Parameters
    ----------
    func : callable
        Function with signature ``f(t) -> float | ndarray``.
    duration : float, optional
        If given, returns a bounded :class:`~climatecritters.core.ForcingElement`
        lasting ``duration`` time units.  If omitted, returns an indefinite
        :class:`~climatecritters.core.Forcing`.

    Returns
    -------
    Forcing or ForcingElement
        * No ``duration`` → :class:`~climatecritters.core.Forcing` (indefinite)
        * With ``duration`` → :class:`~climatecritters.core.ForcingElement` (bounded)

    Examples
    --------
    ```python
    import climatecritters as cc
    from climatecritters.utils.forcing import create_forcing

    # Indefinite: register directly with a model
    f = create_forcing(lambda t: 0.1 * t)
    model.register_forcing('S', f)

    # Bounded: embed in a scenario
    elem = create_forcing(lambda t: 0.1 * t, duration=50.0)
    seq  = cc.forcing.Hold(10, value=0.0) + elem
    model.register_forcing('S', seq.compile())
    ```
    """
    if not callable(func):
        raise TypeError(f"func must be callable; got {type(func)!r}.")
    if duration is None:
        return Forcing(func)
    return ForcingElement(func, float(duration))


# ---------------------------------------------------------------------------
# Named factories (aliases of create_forcing with pre-built lambdas)
# ---------------------------------------------------------------------------

def create_constant_forcing(value, duration=None):
    """Build a constant forcing.

    Parameters
    ----------
    value : float
        The constant value returned for all ``t``.
    duration : float, optional
        If given, returns a :class:`~climatecritters.core.ForcingElement`.
        If omitted, returns an indefinite :class:`~climatecritters.core.Forcing`.

    Returns
    -------
    Forcing or ForcingElement

    Examples
    --------
    ```python
    from climatecritters.utils.forcing import create_constant_forcing

    f = create_constant_forcing(8.0)
    print(f.get_forcing(0.0))    # 8.0
    print(f.get_forcing(999.9))  # 8.0
    ```
    """
    v = float(value)

    def _func(t):
        t_arr = np.asarray(t, dtype=float)
        out = np.full(t_arr.shape, v, dtype=float)
        return float(out) if t_arr.ndim == 0 else out

    return create_forcing(_func, duration=duration)


def create_sinusoid_forcing(A, period, y0=0.0, duration=None):
    """Build a sinusoidal forcing: ``f(t) = y0 + A * sin(2π t / period)``.

    Parameters
    ----------
    A : float
        Amplitude.
    period : float
        Period (same time units as the model).  Must be > 0.
    y0 : float
        Constant offset.  Default 0.0.
    duration : float, optional
        If given, returns a :class:`~climatecritters.core.ForcingElement`.
        If omitted, returns an indefinite :class:`~climatecritters.core.Forcing`.

    Returns
    -------
    Forcing or ForcingElement

    Raises
    ------
    ValueError
        If ``period`` ≤ 0.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    from climatecritters.utils.forcing import create_sinusoid_forcing

    f = create_sinusoid_forcing(A=0.5, period=1.0)
    fig, ax = f.plot(t_span=(0, 2))
    plt.savefig('docs/reference/figures/create_sinusoid_forcing_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    period = float(period)
    if period <= 0.0:
        raise ValueError("period must be > 0.")
    return create_forcing(
        _build_periodic_func([(period, 1.0)], desired_amplitude=float(A), y0=float(y0)),
        duration=duration,
    )


def create_periodic_forcing(periods_powers, desired_amplitude=1, y0=0, duration=None):
    """Build a composite periodic forcing from normalised sine components.

    Component amplitudes are rescaled so their summed peak amplitude equals
    ``desired_amplitude``.

    Parameters
    ----------
    periods_powers : sequence of (float, float)
        Sequence of ``(period, power)`` pairs.  Each ``period`` must be > 0.
        ``power`` sets the relative weight of that component.
    desired_amplitude : float
        Peak amplitude of the composite signal.  Default 1.
    y0 : float
        Constant offset.  Default 0.
    duration : float, optional
        If given, returns a :class:`~climatecritters.core.ForcingElement`.
        If omitted, returns an indefinite :class:`~climatecritters.core.Forcing`.

    Returns
    -------
    Forcing or ForcingElement

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    from climatecritters.utils.forcing import create_periodic_forcing

    # Milankovitch-like: 100 kyr eccentricity + 41 kyr obliquity
    orbital = create_periodic_forcing([(100, 0.6), (41, 0.4)], desired_amplitude=25.0)
    fig, ax = orbital.plot(t_span=(0, 500))
    ax.set_xlabel('time (kyr)'); ax.set_ylabel('forcing (W m⁻²)')
    plt.savefig('docs/reference/figures/create_periodic_forcing_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    return create_forcing(
        _build_periodic_func(periods_powers, desired_amplitude=desired_amplitude, y0=y0),
        duration=duration,
    )


def create_piecewise_forcing(elements, label="forcing"):
    """Build a piecewise forcing from a sequence of :class:`~climatecritters.core.ForcingElement` parts.

    Compiles the sequence immediately and returns a callable
    :class:`~climatecritters.core.Forcing`.

    Parameters
    ----------
    elements : sequence of ForcingElement
        Ordered :class:`~climatecritters.core.Hold`, :class:`~climatecritters.core.Ramp`,
        :class:`~climatecritters.core.Harmonic`, or general
        :class:`~climatecritters.core.ForcingElement` instances.
    label : str
        Human-readable label.  Default ``'forcing'``.

    Returns
    -------
    forcing : Forcing

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    import climatecritters as cc
    from climatecritters.utils.forcing import create_piecewise_forcing

    f = create_piecewise_forcing([
        cc.forcing.Hold(duration=50,  value=0.0),
        cc.forcing.Ramp(duration=100, y0=0.0, yf=4.0),
        cc.forcing.Hold(duration=50,  value=4.0),
    ], label="CO2 ramp")
    fig, ax = f.plot()
    ax.set_xlabel('time'); ax.set_ylabel('forcing')
    plt.savefig('docs/reference/figures/create_piecewise_forcing_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    return ForcingSequence(parts=list(elements), label=label).compile()


# ---------------------------------------------------------------------------
# Reverse bridge: Forcing → ForcingElement
# ---------------------------------------------------------------------------

def make_forcing_element(forcing, duration=None):
    """Convert a :class:`~climatecritters.core.Forcing` into a bounded
    :class:`~climatecritters.core.ForcingElement`.

    This is the reverse of :meth:`~climatecritters.core.ForcingSequence.compile` —
    it lets you embed an existing ``Forcing`` as a timed segment inside a
    :class:`~climatecritters.core.ForcingSequence`.

    Parameters
    ----------
    forcing : Forcing
        The forcing to embed.  Must be callable via ``get_forcing``.
    duration : float, optional
        Length of the segment.  **Required** for lambda/callable-backed
        ``Forcing`` objects.  For array-backed ``Forcing`` objects (created
        from a CSV or data array) the duration is inferred from
        ``time[-1] - time[0]`` if not provided.

    Returns
    -------
    ForcingElement

    Raises
    ------
    ValueError
        If ``duration`` is not provided and cannot be inferred.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    import climatecritters as cc
    from climatecritters.utils.forcing import make_forcing_element

    # Embed empirical insolation data as a bounded segment
    obs  = cc.Forcing.from_csv(dataset='insolation')
    elem = make_forcing_element(obs)          # duration inferred from time axis
    scenario = cc.forcing.Hold(50, value=0.0) + elem + cc.forcing.Hold(50, value=0.0)
    fig, ax = scenario.plot()
    ax.set_xlabel('time (kyr)'); ax.set_ylabel('insolation (W m⁻²)')
    plt.savefig('docs/reference/figures/make_forcing_element_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    if not isinstance(forcing, Forcing):
        raise TypeError(f"forcing must be a Forcing instance; got {type(forcing)!r}.")

    if duration is None:
        # Try to infer from a time axis (array-backed Forcing)
        if forcing.time is not None:
            t = np.asarray(forcing.time, dtype=float)
            duration = float(t[-1] - t[0])
        else:
            raise ValueError(
                "duration must be provided for lambda/callable-backed Forcing objects "
                "(no time axis to infer from)."
            )

    return ForcingElement(forcing.get_forcing, float(duration))
