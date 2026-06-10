"""Core forcing classes for ClimateCritters.

The forcing system has two layers:

**Builder layer** — assemble a piecewise timeline from typed segments:

* :class:`ForcingElement` — base class; also directly instantiable with any callable
* :class:`Hold` — constant segment
* :class:`Ramp` — linear or cosine transition
* :class:`Harmonic` — sinusoidal segment with phase continuity
* :class:`ForcingSequence` — ordered composition; call :meth:`ForcingSequence.compile`
  to produce a ready-to-use :class:`Forcing`

**Signal layer** — a uniform, callable interface consumed by models:

* :class:`Forcing` — wraps a callable, a data array, or a compiled sequence;
  supports value superposition via ``+``

The typical workflow::

    import climatecritters as cc

    # Build a scenario timeline
    seq = cc.Hold(100, value=0.0) + cc.Ramp(50, y0=0.0, yf=4.0) + cc.Hold(100, value=4.0)
    f   = seq.compile()              # → Forcing

    # Register with a model
    model.register_forcing('S', f, attachment_style='additive', timing='pre')

    # Superpose two indefinite signals
    combined = cc.Forcing(orbital_func) + cc.Forcing(noise_func)
    model.register_forcing('S0', combined)
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d


@dataclass(frozen=True)
class ResolvedSegment:
    """Immutable record describing one compiled segment of a :class:`ForcingSequence`.

    Created internally by :meth:`ForcingElement._resolve`; users do not
    construct this directly.

    Attributes
    ----------
    kind : str
        Element kind (``'hold'``, ``'ramp'``, ``'harmonic'``, ``'func'``).
    t0, tf : float
        Absolute start and end times in model time units.
    y0, yf : float
        Forcing value at ``t0`` and ``tf``.
    eval_mode : str
        Evaluation rule (``'constant'``, ``'linear'``, ``'cosine'``,
        ``'harmonic'``, ``'func'``).
    params : dict
        Mode-specific evaluation parameters (e.g. amplitude, omega, callable).
    """

    kind: str
    t0: float
    tf: float
    y0: float
    yf: float
    eval_mode: str
    params: dict


class ForcingElement:
    """A bounded forcing segment backed by an arbitrary callable.

    ``ForcingElement`` serves two roles:

    1. **Base class** for the named segment types :class:`Hold`, :class:`Ramp`,
       and :class:`Harmonic`.  Those subclasses call ``super().__init__`` to
       inherit ``plot_kwargs``.
    2. **Concrete class** for the general case — wrap *any* callable as a
       bounded segment with a fixed duration::

           elem = ForcingElement(lambda t: np.sin(t), duration=10.0)

    Parameters
    ----------
    func : callable
        Function ``f(t) -> float | ndarray``.  Receives the *absolute* time
        value (``t0 + tau``) at each evaluation point.
    duration : float
        Length of the segment in model time units.  Must be > 0.
    plot_kwargs : dict, optional
        Matplotlib keyword arguments used when this element is drawn by
        :meth:`plot` or :meth:`ForcingSequence.plot`.  Common keys:
        ``color``, ``linewidth``, ``linestyle``, ``label``, ``alpha``.
        If ``None`` (default), a colour is chosen automatically by segment
        kind.

    Notes
    -----
    A ``ForcingElement`` is not callable and cannot be used directly as a
    model input.  Compose it into a :class:`ForcingSequence` and call
    :meth:`ForcingSequence.compile`::

        seq = Hold(5, value=0.0) + elem + Hold(5, value=0.0)
        f   = seq.compile()   # → Forcing, ready to register

    **Operator ``+``**

    The behaviour of ``+`` depends on the type of the right-hand operand:

    * ``ForcingElement + ForcingElement`` → :class:`ForcingSequence`
      (temporal concatenation)
    * ``ForcingElement + ForcingSequence`` → :class:`ForcingSequence`
      (prepend to sequence)
    * ``ForcingElement + Forcing`` → :class:`Forcing`
      (additive overlay for the duration of the element; auto-compiles)
    * ``Forcing + ForcingElement`` → :class:`Forcing`
      (same; ``__radd__`` makes this commutative)

    Examples
    --------
    Wrap a custom function as a bounded segment and visualise it:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import climatecritters as cc

    elem = cc.ForcingElement(
        lambda t: np.exp(-0.01 * t), duration=50.0,
        plot_kwargs={'color': 'purple', 'label': 'decay'}
    )
    fig, ax = elem.plot()
    ax.set_title('Exponential decay element')
    plt.savefig('docs/reference/figures/ForcingElement_example.png',
                dpi=150, bbox_inches='tight')
    ```

    Compose into a sequence, then plot the full scenario:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import climatecritters as cc

    elem = cc.ForcingElement(
        lambda t: np.exp(-0.01 * t), duration=50.0,
        plot_kwargs={'color': 'purple', 'label': 'decay'}
    )
    seq = cc.Hold(10, value=1.0) + elem + cc.Hold(10, value=0.0)
    fig, ax = seq.plot()
    plt.savefig('docs/reference/figures/ForcingElement_sequence_example.png',
                dpi=150, bbox_inches='tight')
    ```

    Add centred noise on top of a ramp:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import climatecritters as cc

    rng   = np.random.default_rng(42)
    noise = cc.Forcing(lambda t: rng.normal(0, 0.05))
    noisy_ramp = cc.Ramp(duration=100, y0=0.0, yf=4.0) + noise
    # noisy_ramp is a Forcing lasting 100 time units; t_span required for callable overlay
    fig, ax = noisy_ramp.plot(t_span=(0, 100))
    plt.savefig('docs/reference/figures/ForcingElement_noise_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(self, func=None, duration=None, plot_kwargs=None):
        # Always store plot_kwargs so subclasses that call super().__init__ inherit it.
        self.plot_kwargs = plot_kwargs
        if func is not None:
            if not callable(func):
                raise TypeError(
                    "ForcingElement func must be callable; "
                    f"got {type(func)!r}."
                )
            duration = float(duration)
            if duration <= 0.0:
                raise ValueError("ForcingElement duration must be > 0.")
            self._func = func
            self._duration = duration

    def _resolve(self, t0, y_prev):
        # Only reached for direct ForcingElement instances (not named subclasses).
        if not hasattr(self, "_func"):
            raise NotImplementedError(
                f"{type(self).__name__} must implement _resolve."
            )
        t0 = float(t0)
        tf = t0 + self._duration
        y0 = float(self._func(t0))
        yf = float(self._func(tf))
        return ResolvedSegment(
            kind="func",
            t0=t0,
            tf=tf,
            y0=y0,
            yf=yf,
            eval_mode="func",
            params={"func": self._func, "t0_abs": t0},
        )

    def plot(self, t_span=None, n=300, ax=None, **kwargs):
        """Plot this element over its duration.

        Delegates to :meth:`ForcingSequence.plot` with this element as the
        only part.  ``t_span`` defaults to ``(0, duration)``.

        Parameters
        ----------
        t_span : (float, float), optional
            Time range to plot.  Defaults to ``(0, duration)``.
        n : int
            Number of evaluation points.  Default 300.
        ax : matplotlib.axes.Axes, optional
            Axes to plot into.  A new figure is created if ``None``.
        **kwargs
            Additional keyword arguments passed to ``ax.plot``, overriding
            any ``plot_kwargs`` set on this element.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        return ForcingSequence([self]).plot(t_span=t_span, n=n, ax=ax, **kwargs)

    def __add__(self, other):
        if isinstance(other, ForcingSequence):
            return ForcingSequence([self] + list(other.parts), label=other.label)
        if isinstance(other, ForcingElement):
            return ForcingSequence([self, other])
        if isinstance(other, Forcing):
            # Additive overlay: compile self then superpose with the indefinite Forcing.
            return ForcingSequence([self]).compile() + other
        return NotImplemented

    def __radd__(self, other):
        """Support ``Forcing + ForcingElement`` — additive overlay."""
        if isinstance(other, Forcing):
            return other + self
        return NotImplemented


class Hold(ForcingElement):
    """Constant-valued segment.

    Parameters
    ----------
    value : float
        The constant forcing value throughout the segment.
    duration : float, optional
        Length of the segment.  Alias: ``dt``.  Exactly one of ``duration``,
        ``dt``, or ``tf`` must be provided.
    dt : float, optional
        Alias for ``duration``.
    tf : float, optional
        Absolute end time (used instead of ``duration`` when the end time is
        known but the start time is not yet fixed).
    plot_kwargs : dict, optional
        Matplotlib keyword arguments used when this element is drawn by
        :meth:`~ForcingElement.plot` or :meth:`ForcingSequence.plot`.
        Common keys: ``color``, ``label``, ``linewidth``, ``linestyle``.
        If ``None`` (default), the segment is drawn in the default Hold
        colour (steel blue).

    Examples
    --------
    Standalone preview:

    ```python
    import matplotlib.pyplot as plt
    import climatecritters as cc

    h = cc.Hold(duration=50, value=280.0, plot_kwargs={'color': 'steelblue', 'label': 'baseline'})
    fig, ax = h.plot()
    plt.savefig('docs/reference/figures/Hold_example.png',
                dpi=150, bbox_inches='tight')
    ```

    As part of a labelled scenario:

    ```python
    import matplotlib.pyplot as plt
    import climatecritters as cc

    scenario = (
        cc.Hold(100, value=0.0,   plot_kwargs={'color': 'steelblue', 'label': 'spin-up'})
        + cc.Ramp(50, y0=0.0, yf=4.0, plot_kwargs={'color': 'firebrick', 'label': 'ramp'})
        + cc.Hold(100, value=4.0, plot_kwargs={'color': 'darkorange', 'label': 'perturbed'})
    )
    fig, ax = scenario.plot()
    ax.legend()
    plt.savefig('docs/reference/figures/Hold_scenario_example.png',
                dpi=150, bbox_inches='tight')
    f = scenario.compile()
    ```
    """

    def __init__(self, dt=None, value=None, duration=None, tf=None, plot_kwargs=None):
        super().__init__(plot_kwargs=plot_kwargs)
        if value is None:
            raise ValueError("Hold requires value.")

        provided = [dt is not None, duration is not None, tf is not None]
        if sum(provided) == 0:
            raise ValueError("Hold requires exactly one of dt, duration, or tf.")
        if sum(provided) > 1:
            raise ValueError("Hold accepts only one of dt, duration, or tf.")

        self.value = float(value)
        self.duration = None
        self.tf = None

        if tf is not None:
            self.tf = float(tf)
            return

        use_dt = dt if dt is not None else duration
        self.duration = float(use_dt)
        if self.duration <= 0.0:
            raise ValueError("Hold duration must be > 0.")

    def _resolve(self, t0, y_prev):
        y = self.value
        if self.tf is not None:
            tf = float(np.maximum(float(self.tf), float(t0)))
        else:
            tf = float(t0 + self.duration)
        return ResolvedSegment(
            kind="hold",
            t0=float(t0),
            tf=tf,
            y0=float(y),
            yf=float(y),
            eval_mode="constant",
            params={"value": float(y)},
        )


class Ramp(ForcingElement):
    """Monotonic transition between two values with linear or cosine easing.

    Parameters
    ----------
    duration : float
        Length of the transition.  Alias: ``half_period``.
    half_period : float, optional
        Alias for ``duration`` (legacy name).
    y0 : float, optional
        Starting value.  Alias: ``y_start``.  If omitted, inherits from the
        previous segment's endpoint.
    yf : float, optional
        Ending value.  Alias: ``y_end``.  If omitted, computed from ``A``
        or ``y_exit``.
    A : float, optional
        Signed amplitude of the transition (``yf = y0 + A``).  Used when
        ``yf`` is not specified directly.
    y_exit : float, optional
        Absolute target value combined with ``A`` and ``half_period`` to
        compute a proportionally scaled duration.
    shape : {'linear', 'cosine'}
        Interpolation shape.  ``'cosine'`` gives a smooth S-curve (eased
        start and end); ``'linear'`` is a straight line.  Default
        ``'linear'``.
    plot_kwargs : dict, optional
        Matplotlib keyword arguments used when this element is drawn by
        :meth:`~ForcingElement.plot` or :meth:`ForcingSequence.plot`.
        Common keys: ``color``, ``label``, ``linewidth``, ``linestyle``.
        If ``None`` (default), the segment is drawn in the default Ramp
        colour (orange).

    Notes
    -----
    The ``shape='cosine'`` option applies a half-cosine ease, producing an
    S-curve::

        y(τ) = y0 + (yf - y0) * 0.5 * (1 - cos(π * τ / duration))

    This is smoother than ``'linear'`` at both endpoints and is a good
    choice when the transition is intended to represent a gradual forcing
    change rather than an abrupt one.

    Examples
    --------
    Preview the two shapes side by side:

    ```python
    import matplotlib.pyplot as plt
    import climatecritters as cc

    fig, ax = plt.subplots()
    cc.Ramp(100, y0=0.0, yf=1.0, shape='linear',
            plot_kwargs={'label': 'linear', 'color': 'steelblue'}).plot(ax=ax)
    cc.Ramp(100, y0=0.0, yf=1.0, shape='cosine',
            plot_kwargs={'label': 'cosine', 'color': 'firebrick', 'linestyle': '--'}).plot(ax=ax)
    ax.legend()
    plt.savefig('docs/reference/figures/Ramp_shapes_example.png',
                dpi=150, bbox_inches='tight')
    ```

    CO₂-doubling scenario:

    ```python
    import matplotlib.pyplot as plt
    import climatecritters as cc

    scenario = (
        cc.Hold(200, value=280.0,  plot_kwargs={'color': 'steelblue',  'label': 'pre-industrial'})
        + cc.Ramp(100, y0=280.0, yf=560.0, shape='cosine',
                  plot_kwargs={'color': 'firebrick', 'label': 'CO₂ ramp'})
        + cc.Hold(200, value=560.0, plot_kwargs={'color': 'darkorange', 'label': '2×CO₂'})
    )
    fig, ax = scenario.plot()
    ax.set_xlabel('time (kyr)'); ax.set_ylabel('CO₂ (ppm)'); ax.legend()
    plt.savefig('docs/reference/figures/Ramp_example.png',
                dpi=150, bbox_inches='tight')
    f = scenario.compile()
    ```
    """

    def __init__(
        self,
        half_period=None,
        duration=None,
        y_start=None,
        y_end=None,
        y0=None,
        yf=None,
        A=None,
        y_exit=None,
        shape="linear",
        plot_kwargs=None,
    ):
        super().__init__(plot_kwargs=plot_kwargs)
        if half_period is None and duration is None:
            raise ValueError("Ramp requires half_period or duration.")
        dt = half_period if half_period is not None else duration
        self.half_period = float(dt)
        if self.half_period <= 0.0:
            raise ValueError("Ramp half_period/duration must be > 0.")

        if y0 is None and y_start is not None:
            y0 = y_start
        if yf is None and y_end is not None:
            yf = y_end

        self.y0 = None if y0 is None else float(y0)
        self.yf = None if yf is None else float(yf)
        self.A = None if A is None else float(A)
        self.y_exit = None if y_exit is None else float(y_exit)
        self.shape = str(shape).lower()
        if self.shape not in {"linear", "cosine"}:
            raise ValueError("Ramp shape must be 'linear' or 'cosine'.")

    def _resolve_endpoints_and_duration(self, y_prev):
        y0 = self.y0
        if y0 is None:
            if y_prev is None:
                if self.A is not None and self.y_exit is not None:
                    y0 = float(self.y_exit - self.A)
                else:
                    raise ValueError("Ramp y0 is not specified and no previous segment is available.")
            else:
                y0 = float(y_prev)

        yf = self.yf
        duration_effective = float(self.half_period)
        if yf is None:
            if self.y_exit is not None:
                yf = float(self.y_exit)
                if self.A is not None:
                    if np.isclose(self.A, 0.0):
                        raise ValueError("Ramp A must be non-zero when using y_exit-based duration scaling.")
                    frac = (yf - y0) / self.A
                    if frac < 0.0:
                        raise ValueError("Ramp y_exit implies opposite direction from A.")
                    duration_effective = float(self.half_period * frac)
            elif self.A is not None:
                yf = float(y0 + self.A)
            else:
                raise ValueError("Ramp yf is not specified. Provide yf, or A, or y_exit.")

        duration_effective = max(np.finfo(float).eps, float(duration_effective))
        return float(y0), float(yf), duration_effective

    def _resolve(self, t0, y_prev):
        y0, yf, duration_effective = self._resolve_endpoints_and_duration(y_prev)
        return ResolvedSegment(
            kind="ramp",
            t0=float(t0),
            tf=float(t0 + duration_effective),
            y0=float(y0),
            yf=float(yf),
            eval_mode=self.shape,
            params={"duration_effective": float(duration_effective)},
        )


class Harmonic(ForcingElement):
    """Sinusoidal segment with automatic phase continuity.

    The phase at the start of the segment is computed so that the sinusoid
    passes through ``y0`` (or the previous segment's endpoint), ensuring a
    smooth join when embedded in a :class:`ForcingSequence`.

    Parameters
    ----------
    duration : float
        Length of the segment.  Must be > 0.
    period : float
        Period of the sinusoid in model time units.  Must be > 0.
    A : float
        Amplitude.  Must be non-zero.
    center : float, optional
        Mean value (vertical offset) of the sinusoid.  If omitted, the mean
        is inferred from ``y0`` such that the sinusoid oscillates symmetrically
        about the starting value.
    y0 : float, optional
        Starting value.  If omitted, inherits from the previous segment's
        endpoint.  At least one of ``y0`` or ``center`` must be provided (or
        a previous segment must exist).
    plot_kwargs : dict, optional
        Matplotlib keyword arguments used when this element is drawn by
        :meth:`~ForcingElement.plot` or :meth:`ForcingSequence.plot`.
        Common keys: ``color``, ``label``, ``linewidth``, ``linestyle``.
        If ``None`` (default), the segment is drawn in the default Harmonic
        colour (green).

    Notes
    -----
    ``Harmonic`` is always bounded — it requires a ``duration``.  For a
    perpetual sinusoidal signal (no fixed duration), use
    :func:`~climatecritters.utils.forcing.create_sinusoid_forcing` instead.

    The phase ``φ`` is chosen so that::

        y(t0) = center + A * sin(φ) = y0

    which gives ``φ = arcsin((y0 - center) / A)``.  This guarantees the
    signal is continuous with the preceding segment regardless of where
    in the oscillation cycle the segment begins.

    Examples
    --------
    Preview a standalone harmonic:

    ```python
    import matplotlib.pyplot as plt
    import climatecritters as cc

    h = cc.Harmonic(duration=20, period=4.0, A=0.5, center=0.0,
                    plot_kwargs={'color': 'seagreen', 'label': 'oscillation'})
    fig, ax = h.plot()
    ax.set_title('Harmonic element (20 time units)')
    plt.savefig('docs/reference/figures/Harmonic_example.png',
                dpi=150, bbox_inches='tight')
    ```

    Seasonal oscillation embedded in a scenario — note the smooth join:

    ```python
    import matplotlib.pyplot as plt
    import climatecritters as cc

    scenario = (
        cc.Hold(10, value=0.5, plot_kwargs={'color': 'steelblue',  'label': 'hold'})
        + cc.Harmonic(20, period=4.0, A=0.5, center=0.5,
                      plot_kwargs={'color': 'seagreen', 'label': 'oscillation'})
        + cc.Hold(10, value=0.5, plot_kwargs={'color': 'steelblue'})
    )
    fig, ax = scenario.plot()
    ax.legend()
    plt.savefig('docs/reference/figures/Harmonic_scenario_example.png',
                dpi=150, bbox_inches='tight')
    f = scenario.compile()
    ```
    """

    def __init__(self, duration, period, A, center=None, y0=None, plot_kwargs=None):
        super().__init__(plot_kwargs=plot_kwargs)
        self.duration = float(duration)
        self.period = float(period)
        self.A = float(A)
        self.center = None if center is None else float(center)
        self.y0 = None if y0 is None else float(y0)

        if self.duration <= 0.0:
            raise ValueError("Harmonic duration must be > 0.")
        if self.period <= 0.0:
            raise ValueError("Harmonic period must be > 0.")
        if np.isclose(self.A, 0.0):
            raise ValueError("Harmonic amplitude A must be non-zero.")

    def _resolve(self, t0, y_prev):
        y0 = self.y0
        if y0 is None:
            y0 = y_prev
        if y0 is None and self.center is None:
            raise ValueError("Harmonic requires y0 or center (or previous segment endpoint).")

        c = float(y0 if self.center is None else self.center)
        start = float(y0 if y0 is not None else c)
        arg = (start - c) / self.A
        arg = float(np.clip(arg, -1.0, 1.0))
        phi = float(np.arcsin(arg))
        omega = 2.0 * np.pi / self.period
        y_end = float(c + self.A * np.sin(omega * self.duration + phi))

        return ResolvedSegment(
            kind="harmonic",
            t0=float(t0),
            tf=float(t0 + self.duration),
            y0=float(start),
            yf=float(y_end),
            eval_mode="harmonic",
            params={"center": c, "A": float(self.A), "omega": omega, "phi": phi},
        )


class ForcingSequence:
    """Composable sequence of :class:`ForcingElement` parts.

    ``ForcingSequence`` is a **builder** — it assembles an ordered timeline
    of segments and knows how to resolve them.  It is **not callable** and
    cannot be used directly as a model input.  Call :meth:`compile` to produce
    a :class:`Forcing` that is callable and ready to register::

        seq = Hold(100, value=0.0) + Ramp(50, y0=0.0, yf=4.0) + Hold(100, value=0.0)
        f   = seq.compile()
        model.register_forcing('S', f, attachment_style='additive', timing='pre')

    Parameters
    ----------
    parts : list of ForcingElement, optional
        Initial list of elements.  May also be built incrementally with ``+``.
    label : str
        Human-readable label carried through to the compiled :class:`Forcing`.
        Default ``'forcing'``.

    Notes
    -----
    **Operator ``+``**

    * ``ForcingSequence + ForcingElement`` → :class:`ForcingSequence`
      (append element)
    * ``ForcingSequence + ForcingSequence`` → :class:`ForcingSequence`
      (concatenate)
    * ``ForcingSequence + Forcing`` → :class:`Forcing`
      (additive overlay for the full duration of the sequence; auto-compiles)
    * ``Forcing + ForcingSequence`` → :class:`Forcing`
      (same; ``__radd__`` makes this commutative)

    **No memoization** — :meth:`compile` always produces a fresh
    :class:`Forcing`.  Recompile freely after modifying ``parts``.

    **Visualisation** — call :meth:`plot` to inspect the scenario before
    registering it.  Each segment is colour-coded by type and labelled from
    its ``plot_kwargs`` if provided.

    Examples
    --------
    Build, visualise, and register a CO₂-ramp scenario:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import climatecritters as cc

    scenario = (
        cc.Hold(200, value=280.0,
                plot_kwargs={'color': 'steelblue',  'label': 'pre-industrial'})
        + cc.Ramp(100, y0=280.0, yf=560.0, shape='cosine',
                  plot_kwargs={'color': 'firebrick', 'label': 'CO₂ ramp'})
        + cc.Hold(200, value=560.0,
                  plot_kwargs={'color': 'darkorange', 'label': '2×CO₂'})
    )

    # Inspect before committing
    fig, ax = scenario.plot()
    ax.set_xlabel('time (kyr)'); ax.set_ylabel('CO₂ (ppm)'); ax.legend()
    plt.savefig('docs/reference/figures/ForcingSequence_example.png',
                dpi=150, bbox_inches='tight')

    # Overlay noise to produce a bounded Forcing over the same 500-unit window
    rng   = np.random.default_rng(0)
    noise = cc.Forcing(lambda t: rng.normal(0, 5.0))
    f     = scenario + noise   # → Forcing; 500-unit bounded signal + noise
    ```
    """

    def __init__(self, parts=None, label="forcing"):
        self.parts = [] if parts is None else list(parts)
        self.label = str(label)

    def __add__(self, other):
        if isinstance(other, ForcingElement):
            return ForcingSequence(self.parts + [other], label=self.label)
        if isinstance(other, ForcingSequence):
            return ForcingSequence(self.parts + other.parts, label=self.label)
        if isinstance(other, Forcing):
            # Additive overlay: compile self then superpose with the indefinite Forcing.
            return self.compile() + other
        return NotImplemented

    def __radd__(self, other):
        """Support ``Forcing + ForcingSequence`` — additive overlay."""
        if isinstance(other, Forcing):
            return other + self
        return NotImplemented

    def plot(self, t_span=None, n=300, ax=None, **kwargs):
        """Plot the sequence, colour-coded by segment type.

        Each segment is drawn separately using its own ``plot_kwargs`` if set,
        otherwise a default colour is chosen by segment kind (``Hold`` →
        blue, ``Ramp`` → orange, ``Harmonic`` → green, callable → red).
        Vertical dotted lines mark the transitions between segments.

        Parameters
        ----------
        t_span : (float, float), optional
            Time range to plot.  Defaults to ``(0, t_end)`` of the compiled
            sequence.
        n : int
            Total number of evaluation points distributed proportionally
            across segments.  Default 300.
        ax : matplotlib.axes.Axes, optional
            Axes to plot into.  A new figure is created if ``None``.
        **kwargs
            Additional keyword arguments applied to **all** segments,
            overriding per-element ``plot_kwargs``.  Useful for e.g.
            ``linewidth`` or ``alpha``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        ```python
        import matplotlib.pyplot as plt
        import climatecritters as cc

        scenario = (
            cc.Hold(100, value=0.0, plot_kwargs={'color': 'steelblue', 'label': 'baseline'})
            + cc.Ramp(50, y0=0.0, yf=4.0, plot_kwargs={'color': 'firebrick', 'label': 'ramp'})
            + cc.Hold(100, value=4.0, plot_kwargs={'color': 'darkorange', 'label': 'perturbed'})
        )
        fig, ax = scenario.plot()
        ax.set_xlabel('time'); ax.set_ylabel('forcing'); ax.legend()
        plt.savefig('docs/reference/figures/ForcingSequence_plot_example.png',
                    dpi=150, bbox_inches='tight')
        ```
        """
        import matplotlib.pyplot as plt

        _DEFAULT_COLORS = {
            'hold':     'steelblue',
            'ramp':     'darkorange',
            'harmonic': 'seagreen',
            'func':     'firebrick',
        }

        # Resolve segments to get absolute t0/tf per part.
        resolved = []
        t_cursor = 0.0
        y_prev = None
        for part in self.parts:
            seg = part._resolve(t_cursor, y_prev)
            resolved.append((part, seg))
            t_cursor = float(seg.tf)
            y_prev = float(seg.yf)

        t_end = resolved[-1][1].tf
        compiled = self.compile()

        if t_span is None:
            t_span = (0.0, t_end)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        total_dur = max(t_span[1] - t_span[0], 1e-12)
        legend_labels = set()

        for part, seg in resolved:
            # Clip segment to t_span
            t0_plot = max(float(seg.t0), t_span[0])
            tf_plot = min(float(seg.tf), t_span[1])
            if t0_plot >= tf_plot:
                continue

            seg_dur = float(seg.tf) - float(seg.t0)
            n_seg = max(2, int(n * seg_dur / total_dur))
            t_vals = np.linspace(t0_plot, tf_plot, n_seg)
            y_vals = np.array([compiled.get_forcing(t) for t in t_vals])

            # Build plot style: element's plot_kwargs → default by kind → caller kwargs
            seg_style = {'color': _DEFAULT_COLORS.get(seg.kind, 'gray')}
            if part.plot_kwargs is not None:
                seg_style.update(part.plot_kwargs)
            seg_style.update(kwargs)

            # Avoid duplicate legend entries
            label = seg_style.get('label')
            if label is not None and label in legend_labels:
                seg_style.pop('label')
            elif label is not None:
                legend_labels.add(label)

            ax.plot(t_vals, y_vals, **seg_style)

        # Vertical dotted lines at internal transitions
        for _, seg in resolved[:-1]:
            if t_span[0] < seg.tf < t_span[1]:
                ax.axvline(float(seg.tf), color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

        ax.set_xlabel('time')
        ax.set_ylabel('forcing')
        return fig, ax

    def compile(self):
        """Resolve all elements and return a callable :class:`Forcing`.

        Each call produces a **fresh** :class:`Forcing` with no internal
        caching.  The sequence itself is unchanged and can be extended and
        recompiled freely::

            seq = Hold(50, value=0.0) + Ramp(100, y0=0.0, yf=4.0)
            f1  = seq.compile()
            seq = seq + Hold(50, value=4.0)
            f2  = seq.compile()   # f1 is unaffected

        Returns
        -------
        forcing : Forcing
            A bounded :class:`Forcing` defined over ``[0, t_end]``.  Outside
            this interval the value is held at the first / last segment's
            boundary value.

        Raises
        ------
        ValueError
            If the sequence has no parts.
        TypeError
            If any part is not a :class:`ForcingElement`.
        """
        if len(self.parts) == 0:
            raise ValueError("ForcingSequence has no parts.")

        segs = []
        t_cursor = 0.0
        y_prev = None
        for part in self.parts:
            if not isinstance(part, ForcingElement):
                raise TypeError(f"Unsupported forcing part type: {type(part)!r}")
            seg = part._resolve(t_cursor, y_prev)
            segs.append(seg)
            t_cursor = float(seg.tf)
            y_prev = float(seg.yf)

        marks = [float(seg.tf) for seg in segs[:-1]]
        transition_times = np.array(sorted(set(np.round(marks, 12))), dtype=float)

        compiled = {
            "label": self.label,
            "segments": segs,
            "t_end": float(segs[-1].tf),
            "y_start": float(segs[0].y0),
            "y_end": float(segs[-1].yf),
            "transition_times": transition_times,
            "n_transitions": int(transition_times.size),
        }
        return Forcing._from_compiled(compiled)


class Forcing:
    """Unified time-varying signal consumed by models.

    ``Forcing`` wraps any time-dependent input and exposes a single
    ``get_forcing(t)`` interface.  It accepts four kinds of input:

    **1. A callable** (lambda, function, or any object with ``__call__``)::

        f = Forcing(lambda t: 1360.0 + 5.0 * np.sin(2 * np.pi * t / 11.0))

    **2. A data array** with an optional time axis (interpolated)::

        f = Forcing(data=values, time=t_axis, interpolation='cubic')

    **3. A CSV dataset** (two built-in datasets are bundled)::

        f = Forcing.from_csv(dataset='vieira_tsi')
        f = Forcing.from_csv(dataset='insolation')
        f = Forcing.from_csv(file_path='my_data.csv', time_name='age', value_name='co2')

    **4. A** :class:`ForcingSequence` **(auto-compiled)**::

        seq = Hold(100, value=0.0) + Ramp(50, y0=0.0, yf=4.0) + Hold(100, value=4.0)
        f   = Forcing(seq)   # equivalent to seq.compile()

    Parameters
    ----------
    data : callable, array-like, or ForcingSequence
        The signal source.  See above for accepted types.
    time : array-like, optional
        Time axis for array-backed ``Forcing``.  If omitted, integer indices
        are used.  Must be strictly increasing and the same length as ``data``.
    params : dict, optional
        Extra keyword arguments forwarded to a callable ``data`` via
        ``functools.partial``.
    interpolation : {'cubic', 'linear'}
        Interpolation method for array-backed ``Forcing``.  Default ``'cubic'``
        (extrapolates outside the data range).

    Attributes
    ----------
    forcing_type : str
        One of ``'function'``, ``'sequence'``, ``'interpolated array cubic'``,
        or ``'interpolated array linear'``.
    summary : dict or None
        For sequence-backed ``Forcing``, a summary dict with keys ``label``,
        ``t_end``, ``y_start``, ``y_end``, ``n_parts``, ``n_transitions``,
        ``transition_times``.  ``None`` for other types.

    Notes
    -----
    **Value superposition with ``+``**

    ``Forcing`` objects can be combined with ``+`` to produce a new
    ``Forcing`` whose value is the sum of both operands at each ``t``.
    The unifying rule is:

    * Both operands indefinite → result indefinite:
      ``Forcing + Forcing``, ``Forcing + callable``
    * One operand bounded (:class:`ForcingElement` or :class:`ForcingSequence`)
      → result bounded for the duration of the bounded operand:
      ``Forcing + ForcingElement``, ``ForcingSequence + Forcing``, etc.

    The bounded operand is auto-compiled before superposition.  Outside its
    interval the compiled ``Forcing`` holds at its boundary value, so the
    sum is well-defined for all ``t``.

    Full operator table:

    .. list-table::
       :header-rows: 1

       * - Left
         - Right
         - Result
         - Semantic
       * - ``Forcing``
         - ``Forcing``
         - ``Forcing`` (indefinite)
         - value superposition for all t
       * - ``Forcing``
         - callable
         - ``Forcing`` (indefinite)
         - value superposition for all t
       * - ``ForcingElement``
         - ``ForcingElement``
         - ``ForcingSequence``
         - temporal concatenation
       * - ``ForcingSequence``
         - ``ForcingElement`` / ``ForcingSequence``
         - ``ForcingSequence``
         - temporal concatenation
       * - ``ForcingElement`` / ``ForcingSequence``
         - ``Forcing``
         - ``Forcing`` (bounded)
         - additive overlay; auto-compiles
       * - ``Forcing``
         - ``ForcingElement`` / ``ForcingSequence``
         - ``Forcing`` (bounded)
         - additive overlay; auto-compiles

    Examples
    --------
    Callable-backed (perpetual):

    ```python
    import numpy as np
    import climatecritters as cc

    orbital = cc.Forcing(lambda t: 5.0 * np.sin(2 * np.pi * t / 100.0))
    model.register_forcing('S0', orbital)
    ```

    Array-backed from real data:

    ```python
    insol = cc.Forcing.from_csv(dataset='insolation')
    model.register_forcing('S0', insol)
    ```

    Piecewise scenario:

    ```python
    scenario = (
        cc.Hold(duration=200, value=280.0)
        + cc.Ramp(duration=100, y0=280.0, yf=560.0, shape='cosine')
        + cc.Hold(duration=200, value=560.0)
    )
    f = scenario.compile()
    model.register_forcing('CO2', f)
    ```

    Superpose orbital and noise:

    ```python
    rng   = np.random.default_rng(0)
    noise = cc.Forcing(lambda t: rng.normal(0, 0.5))
    combined = orbital + noise
    model.register_forcing('S0', combined)
    ```

    Add noise to a bounded scenario:

    ```python
    noisy = scenario + noise   # → Forcing, 500 time-unit duration
    model.register_forcing('CO2', noisy)
    ```

    Visualise before registering:

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import climatecritters as cc

    scenario = (
        cc.Hold(200, value=280.0)
        + cc.Ramp(100, y0=280.0, yf=560.0, shape='cosine')
        + cc.Hold(200, value=560.0)
    )
    orbital = cc.Forcing(lambda t: 5.0 * np.sin(2 * np.pi * t / 100.0))

    # Sequence-backed — t_span inferred from summary
    fig, ax = scenario.compile().plot()
    plt.savefig('docs/reference/figures/Forcing_sequence_example.png',
                dpi=150, bbox_inches='tight')

    # Callable-backed — t_span required
    fig, ax = orbital.plot(t_span=(0, 500))
    plt.savefig('docs/reference/figures/Forcing_callable_example.png',
                dpi=150, bbox_inches='tight')

    # Array-backed — t_span inferred from time axis
    fig, ax = cc.Forcing.from_csv(dataset='insolation').plot()
    plt.savefig('docs/reference/figures/Forcing_array_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    _INTERPOLATION_KINDS = {"cubic", "linear"}

    def __init__(self, data, time=None, params=None, interpolation="cubic"):
        self.data = data
        self.time = time
        self.params = {} if params is None else dict(params)
        self.forcing_type = None
        self.summary = None

        if isinstance(data, ForcingSequence):
            # Delegate to compile() which returns a fully initialised Forcing.
            # Copy internals rather than wrapping again to avoid recursion.
            compiled_forcing = data.compile()
            self.forcing_type = compiled_forcing.forcing_type
            self.forcing_func = compiled_forcing.forcing_func
            self.summary = compiled_forcing.summary
            return

        if isinstance(data, ForcingElement):
            compiled_forcing = ForcingSequence([data]).compile()
            self.forcing_type = compiled_forcing.forcing_type
            self.forcing_func = compiled_forcing.forcing_func
            self.summary = compiled_forcing.summary
            return

        if callable(data):
            self.forcing_type = "function"
            self.forcing_func = functools.partial(data, **self.params)
            return

        self._init_interpolated_array(data=data, time=time, params=self.params, interpolation=interpolation)

    def _init_interpolated_array(self, data, time, params, interpolation):
        interpolation = str(interpolation).lower()
        if interpolation not in self._INTERPOLATION_KINDS:
            valid = ", ".join(sorted(self._INTERPOLATION_KINDS))
            raise ValueError(f"Unsupported interpolation '{interpolation}'. Valid options: {valid}.")

        values = np.asarray(data, dtype=float)
        if values.ndim != 1:
            raise ValueError("Forcing array data must be one-dimensional.")
        if values.size == 0:
            raise ValueError("Forcing array data must not be empty.")

        if time is None:
            t_axis = np.arange(values.size, dtype=float)
        else:
            t_axis = np.asarray(time, dtype=float)
            if t_axis.shape != values.shape:
                raise ValueError("Forcing time axis must have the same shape as data.")

        if not np.all(np.isfinite(t_axis)) or not np.all(np.isfinite(values)):
            raise ValueError("Forcing time and data must be finite.")
        if np.any(np.diff(t_axis) <= 0):
            raise ValueError("Forcing time axis must be strictly increasing.")

        self.data = values
        self.time = t_axis
        self.forcing_type = f"interpolated array {interpolation}"

        if values.size == 1:
            const_val = float(values[0])

            def _constant_interp(t):
                t_arr = np.asarray(t, dtype=float)
                if t_arr.ndim == 0:
                    return const_val
                return np.full(t_arr.shape, const_val, dtype=float)

            self.forcing_func = _constant_interp
            return

        interp_kwargs = dict(params)
        if interpolation == "cubic":
            interp_kwargs.setdefault("extrapolate", True)
            self.forcing_func = CubicSpline(t_axis, values, **interp_kwargs)
        else:
            interp_kwargs.setdefault("kind", "linear")
            interp_kwargs.setdefault("bounds_error", False)
            interp_kwargs.setdefault("fill_value", (float(values[0]), float(values[-1])))
            self.forcing_func = interp1d(t_axis, values, **interp_kwargs)

    @classmethod
    def _from_compiled(cls, compiled: dict):
        """Internal factory: build a Forcing directly from a compiled segment dict.

        Called by :meth:`ForcingSequence.compile`; avoids going back through
        ``__init__`` which would recurse into the ForcingSequence path.
        """
        obj = cls.__new__(cls)
        obj.data = None
        obj.time = None
        obj.params = {}
        obj.forcing_type = "sequence"

        segs = compiled["segments"]
        t_end = compiled["t_end"]
        y_end = compiled["y_end"]

        def _eval_scalar(t_scalar):
            t = float(t_scalar)
            if t <= 0.0:
                return float(segs[0].y0)
            if t >= t_end:
                return float(y_end)
            for seg in segs:
                if t <= seg.tf + 1e-12:
                    tau = t - seg.t0
                    dur = seg.tf - seg.t0
                    if seg.eval_mode == "constant":
                        return float(seg.params["value"])
                    if seg.eval_mode == "linear":
                        if dur <= 0.0:
                            return float(seg.yf)
                        return float(seg.y0 + (seg.yf - seg.y0) * tau / dur)
                    if seg.eval_mode == "cosine":
                        if dur <= 0.0:
                            return float(seg.yf)
                        frac = np.clip(tau / dur, 0.0, 1.0)
                        return float(seg.y0 + (seg.yf - seg.y0) * 0.5 * (1.0 - np.cos(np.pi * frac)))
                    if seg.eval_mode == "harmonic":
                        p = seg.params
                        return float(p["center"] + p["A"] * np.sin(p["omega"] * tau + p["phi"]))
                    if seg.eval_mode == "func":
                        return float(seg.params["func"](seg.params["t0_abs"] + tau))
                    raise ValueError(f"Unsupported eval mode {seg.eval_mode!r}")
            return float(y_end)

        def _forcing_func(t):
            t_arr = np.asarray(t, dtype=float)
            scalar = t_arr.ndim == 0
            t_flat = t_arr.reshape(-1)
            y = np.array([_eval_scalar(tt) for tt in t_flat], dtype=float)
            if scalar:
                return float(y[0])
            return y.reshape(t_arr.shape)

        obj.forcing_func = _forcing_func
        obj.summary = {
            "label": compiled["label"],
            "t_end": t_end,
            "y_start": compiled["y_start"],
            "y_end": y_end,
            "n_parts": len(segs),
            "n_transitions": compiled["n_transitions"],
            "transition_times": compiled["transition_times"].copy(),
        }
        return obj

    @classmethod
    def from_sequence(cls, parts: Iterable[ForcingElement], label="forcing"):
        """Build a :class:`Forcing` from an iterable of :class:`ForcingElement` parts.

        Convenience alias for ``ForcingSequence(parts, label).compile()``.

        Parameters
        ----------
        parts : iterable of ForcingElement
            Ordered :class:`Hold`, :class:`Ramp`, :class:`Harmonic`, or general
            :class:`ForcingElement` instances defining the piecewise timeline.
        label : str
            Human-readable label for the resulting forcing object.

        Returns
        -------
        forcing : Forcing

        Examples
        --------
        ```python
        import climatecritters as cc

        f = cc.Forcing.from_sequence([
            cc.Hold(duration=100, value=0.0),
            cc.Ramp(duration=50, y0=0.0, yf=1.0, shape='cosine'),
            cc.Hold(duration=100, value=1.0),
        ], label='freshwater_hosing')
        ```
        """
        return ForcingSequence(parts=list(parts), label=label).compile()

    @classmethod
    def from_csv(
        cls,
        dataset=None,
        file_path=None,
        value_name=None,
        time_name=None,
        params=None,
        interpolation="cubic",
    ):
        """Build a :class:`Forcing` from a CSV file or a bundled dataset.

        Two datasets are included with ClimateCritters:

        * ``'vieira_tsi'`` — Vieira et al. total solar irradiance reconstruction
        * ``'insolation'`` — 65°N summer solstice insolation (Laskar et al.)

        Parameters
        ----------
        dataset : {'vieira_tsi', 'insolation'}, optional
            Name of a bundled dataset.  Mutually exclusive with ``file_path``.
        file_path : str or Path, optional
            Path to an arbitrary CSV file.  Mutually exclusive with ``dataset``.
        value_name : str, optional
            Column name for the forcing values.  If omitted, the first column
            (or the dataset default) is used.
        time_name : str, optional
            Column name for the time axis.  If omitted, integer indices are used
            (or the dataset default is applied).
        params : dict, optional
            Extra keyword arguments forwarded to the interpolator.
        interpolation : {'cubic', 'linear'}
            Interpolation method.  Default ``'cubic'``.

        Returns
        -------
        forcing : Forcing

        Examples
        --------
        ```python
        import climatecritters as cc

        # Bundled dataset
        tsi = cc.Forcing.from_csv(dataset='vieira_tsi')

        # Custom CSV
        co2 = cc.Forcing.from_csv(
            file_path='data/co2_record.csv',
            time_name='age_kyr',
            value_name='co2_ppm',
            interpolation='linear',
        )
        ```
        """
        if dataset is not None:
            if dataset == "vieira_tsi":
                my_resources = importlib.resources.files("climatecritters") / "data"
                file_path = my_resources.joinpath("vieira_tsi.csv")
                default_time = "Age (yrs BP)"
                default_value = "0"
            elif dataset == "insolation":
                my_resources = importlib.resources.files("climatecritters") / "data"
                file_path = my_resources.joinpath("insolation.csv")
                default_time = "kyear"
                default_value = "insol_65N_d172_centered"
            else:
                raise ValueError("Dataset not recognized. Supported datasets are 'vieira_tsi' and 'insolation'.")
            if time_name is None:
                time_name = default_time
            if value_name is None:
                value_name = default_value
        elif file_path is None:
            raise ValueError("Provide either dataset or file_path.")

        csv_path = Path(file_path)
        df = pd.read_csv(csv_path)

        if value_name is None:
            value_name = df.columns[0]
        if value_name not in df.columns:
            raise ValueError(f"Column '{value_name}' not found in {csv_path}.")

        data = df[value_name].to_numpy(dtype=float)

        if time_name is None:
            time = np.arange(len(data), dtype=float)
        else:
            if time_name not in df.columns:
                raise ValueError(f"Column '{time_name}' not found in {csv_path}.")
            time = df[time_name].to_numpy(dtype=float)

        return cls(data=data, time=time, params=params, interpolation=interpolation)

    def get_forcing(self, t):
        """Return the forcing value at time ``t``.

        Parameters
        ----------
        t : float or array-like
            Time value(s).  Accepts scalars or NumPy arrays; returns the same
            shape.

        Returns
        -------
        float or ndarray
        """
        return self.forcing_func(t)

    def plot(self, t_span=None, n=300, ax=None, **kwargs):
        """Plot the forcing signal over a time range.

        Parameters
        ----------
        t_span : (float, float), optional
            Time range ``(t0, tf)`` to evaluate and plot.  For
            sequence-backed ``Forcing`` objects this defaults to
            ``(0, summary["t_end"])``.  For array-backed objects it
            defaults to ``(time[0], time[-1])``.  For callable-backed
            objects ``t_span`` is **required**.
        n : int
            Number of evaluation points.  Default 300.
        ax : matplotlib.axes.Axes, optional
            Axes to plot into.  A new figure is created if ``None``.
        **kwargs
            Additional keyword arguments forwarded to ``ax.plot``
            (e.g. ``color``, ``linewidth``, ``label``).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Raises
        ------
        ValueError
            If ``t_span`` is not provided for a callable-backed
            ``Forcing`` that has no time axis.

        Examples
        --------
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        import climatecritters as cc

        # Sequence-backed — t_span inferred automatically
        f = cc.Forcing.from_sequence([
            cc.Hold(100, value=0.0),
            cc.Ramp(50, y0=0.0, yf=4.0),
        ])
        fig, ax = f.plot()
        plt.savefig('docs/reference/figures/Forcing_plot_example.png',
                    dpi=150, bbox_inches='tight')

        # Callable-backed — t_span required
        solar = cc.Forcing(lambda t: 1360.0 + 5.0 * np.sin(t))
        fig, ax = solar.plot(t_span=(0, 50))
        plt.savefig('docs/reference/figures/Forcing_plot_callable_example.png',
                    dpi=150, bbox_inches='tight')
        ```
        """
        import matplotlib.pyplot as plt

        if t_span is None:
            if self.summary is not None:
                t_span = (0.0, float(self.summary['t_end']))
            elif self.time is not None:
                t_arr = np.asarray(self.time, dtype=float)
                t_span = (float(t_arr[0]), float(t_arr[-1]))
            else:
                raise ValueError(
                    "t_span is required for callable-backed Forcing objects "
                    "(no time axis or sequence summary to infer from)."
                )

        t_vals = np.linspace(float(t_span[0]), float(t_span[1]), int(n))
        y_vals = self.forcing_func(t_vals)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(t_vals, y_vals, **kwargs)
        ax.set_xlabel('time')
        ax.set_ylabel('forcing')
        return fig, ax

    def __add__(self, other):
        """Value superposition — return a new :class:`Forcing` equal to ``self(t) + other(t)``.

        Parameters
        ----------
        other : Forcing, ForcingElement, ForcingSequence, or callable

        Returns
        -------
        Forcing
            Bounded if ``other`` is a :class:`ForcingElement` or
            :class:`ForcingSequence` (duration inherited from the bounded
            operand, holding at boundary values outside the interval).
            Indefinite otherwise.

        Notes
        -----
        ``ForcingElement`` and ``ForcingSequence`` operands are auto-compiled
        before superposition.
        """
        if isinstance(other, (ForcingElement, ForcingSequence)):
            if isinstance(other, ForcingElement):
                other = ForcingSequence([other]).compile()
            else:
                other = other.compile()
        if isinstance(other, Forcing):
            f_a = self.forcing_func
            f_b = other.forcing_func
            def _sum(t):
                return f_a(t) + f_b(t)
            return Forcing(_sum)
        if callable(other):
            f_a = self.forcing_func
            def _sum_callable(t):
                return f_a(t) + other(t)
            return Forcing(_sum_callable)
        return NotImplemented

    def __radd__(self, other):
        """Support ``callable + Forcing`` and ``ForcingElement/Sequence + Forcing``."""
        if isinstance(other, (ForcingElement, ForcingSequence, Forcing)):
            return self.__add__(other)
        if callable(other):
            f_b = self.forcing_func
            def _sum_callable(t):
                return other(t) + f_b(t)
            return Forcing(_sum_callable)
        return NotImplemented


_VALID_ATTACHMENT_STYLES = {"replacement", "additive"}
_VALID_TIMINGS = {"pre", "post"}


@dataclass
class ForcingSpec:
    """Registration record for a single forcing attachment on a model variable.

    Created internally by :meth:`~climatecritters.core.CCModel.register_forcing`;
    users do not construct this directly.

    Parameters
    ----------
    forcing_object : Forcing, callable, or object with get_forcing
        The signal source.  Accepts a :class:`Forcing` instance, any callable
        ``f(t)``, or any object with a ``get_forcing(t)`` method.
    attachment_style : {'replacement', 'additive'}
        How the forcing value is applied to the target variable.

        * ``'replacement'`` — substitutes the current parameter or state value
          entirely at each step.
        * ``'additive'`` — adds the forcing value on top of the existing value
          (for parameters) or to ``dx/dt`` / ``x`` (for state variables).
    timing : {'pre', 'post'}
        When the forcing is applied relative to each integration step.

        * ``'pre'`` — applied inside the RHS function before the integrator
          advances.  The only valid timing for parameters.
        * ``'post'`` — applied as a correction after the integrator step.
          Required for state-variable replacement; also valid for additive state.

    Notes
    -----
    Validation rules (enforced by :meth:`~climatecritters.core.CCModel.register_forcing`,
    not here):

    * parameter + replacement → timing forced to ``'pre'``
    * parameter + additive    → timing forced to ``'pre'``
    * state + replacement     → timing forced to ``'post'``
    * state + additive        → timing required from caller
    """

    forcing_object: object
    attachment_style: str
    timing: str

    def __post_init__(self):
        if self.attachment_style not in _VALID_ATTACHMENT_STYLES:
            valid = ", ".join(sorted(_VALID_ATTACHMENT_STYLES))
            raise ValueError(
                f"attachment_style must be one of {{{valid}}}; "
                f"got {self.attachment_style!r}."
            )
        if self.timing not in _VALID_TIMINGS:
            valid = ", ".join(sorted(_VALID_TIMINGS))
            raise ValueError(
                f"timing must be one of {{{valid}}}; got {self.timing!r}."
            )
        if not (
            callable(self.forcing_object)
            or hasattr(self.forcing_object, "get_forcing")
        ):
            raise TypeError(
                "forcing_object must be callable or have a get_forcing method."
            )

    def evaluate(self, t):
        """Return the forcing value at time ``t``.

        Delegates to ``get_forcing(t)`` if available, otherwise calls the
        object directly.
        """
        if hasattr(self.forcing_object, "get_forcing"):
            return self.forcing_object.get_forcing(t)
        return self.forcing_object(t)


__all__ = [
    "ResolvedSegment",
    "ForcingElement",
    "Hold",
    "Ramp",
    "Harmonic",
    "ForcingSequence",
    "Forcing",
    "ForcingSpec",
]
