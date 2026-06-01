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
    kind: str
    t0: float
    tf: float
    y0: float
    yf: float
    eval_mode: str
    params: dict


class ForcingElement:
    """Base class for composable forcing elements."""

    def __add__(self, other):
        if isinstance(other, ForcingSequence):
            return ForcingSequence([self] + list(other.parts), label=other.label)
        if isinstance(other, ForcingElement):
            return ForcingSequence([self, other])
        return NotImplemented


class Hold(ForcingElement):
    """Constant segment over a duration or until absolute tf."""

    def __init__(self, dt=None, value=None, duration=None, tf=None):
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
    """Monotonic transition over a duration with linear or cosine easing."""

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
    ):
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
    """Sinusoidal segment constrained by duration, period, and amplitude."""

    def __init__(self, duration, period, A, center=None, y0=None):
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
    """Composable sequence of forcing elements."""

    def __init__(self, parts=None, label="forcing"):
        self.parts = [] if parts is None else list(parts)
        self.label = str(label)
        self._compiled = None

    def __add__(self, other):
        if isinstance(other, ForcingElement):
            return ForcingSequence(self.parts + [other], label=self.label)
        if isinstance(other, ForcingSequence):
            return ForcingSequence(self.parts + other.parts, label=self.label)
        return NotImplemented

    def compile(self):
        if self._compiled is not None:
            return self._compiled
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

        self._compiled = {
            "label": self.label,
            "segments": segs,
            "t_end": float(segs[-1].tf),
            "y_start": float(segs[0].y0),
            "y_end": float(segs[-1].yf),
            "transition_times": transition_times,
            "n_transitions": int(transition_times.size),
        }
        return self._compiled

    def summary(self):
        c = self.compile()
        return {
            "label": c["label"],
            "t_end": c["t_end"],
            "y_start": c["y_start"],
            "y_end": c["y_end"],
            "n_parts": len(c["segments"]),
            "n_transitions": c["n_transitions"],
            "transition_times": c["transition_times"].copy(),
        }

    def _eval_scalar(self, t_scalar):
        c = self.compile()
        t = float(t_scalar)
        if t <= 0.0:
            return c["segments"][0].y0
        if t >= c["t_end"]:
            return c["segments"][-1].yf

        for seg in c["segments"]:
            if t <= seg.tf + 1e-12:
                tau = t - seg.t0
                dur = seg.tf - seg.t0
                if seg.eval_mode == "constant":
                    return float(seg.params["value"])
                if seg.eval_mode == "linear":
                    if dur <= 0.0:
                        return float(seg.yf)
                    frac = tau / dur
                    return float(seg.y0 + (seg.yf - seg.y0) * frac)
                if seg.eval_mode == "cosine":
                    if dur <= 0.0:
                        return float(seg.yf)
                    frac = np.clip(tau / dur, 0.0, 1.0)
                    return float(seg.y0 + (seg.yf - seg.y0) * 0.5 * (1.0 - np.cos(np.pi * frac)))
                if seg.eval_mode == "harmonic":
                    p = seg.params
                    return float(p["center"] + p["A"] * np.sin(p["omega"] * tau + p["phi"]))
                raise ValueError(f"Unsupported eval mode {seg.eval_mode!r}")
        return float(c["y_end"])

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_flat = t_arr.reshape(-1)
        y = np.array([self._eval_scalar(tt) for tt in t_flat], dtype=float)
        if scalar:
            return float(y[0])
        return y.reshape(t_arr.shape)


class Forcing:
    """Unified forcing object for callables, time series arrays, and element sequences."""

    _INTERPOLATION_KINDS = {"cubic", "linear"}

    def __init__(self, data, time=None, params=None, interpolation="cubic"):
        self.data = data
        self.time = time
        self.params = {} if params is None else dict(params)
        self.forcing_type = None
        self.summary = None

        if isinstance(data, ForcingSequence):
            data.compile()
            self.forcing_type = "sequence"
            self.forcing_func = data
            self.summary = data.summary()
            return

        if isinstance(data, ForcingElement):
            sequence = ForcingSequence([data])
            sequence.compile()
            self.forcing_type = "sequence"
            self.forcing_func = sequence
            self.summary = sequence.summary()
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
    def from_sequence(cls, parts: Iterable[ForcingElement], label="forcing"):
        sequence = ForcingSequence(parts=list(parts), label=label)
        return cls(data=sequence)

    @classmethod
    def from_elements(cls, elements, y0, label="forcing"):
        if elements is None:
            raise ValueError("elements must not be None.")

        parts = []
        current = float(y0)

        for element in list(elements):
            if isinstance(element, ForcingElement):
                parts.append(element)
                current = float(element._resolve(0.0, current).yf)
                continue

            if not isinstance(element, dict):
                raise TypeError("Forcing elements must be ForcingElement instances or dictionaries.")

            kind = str(element.get("kind", "")).lower()
            if kind in {"constant", "hold"}:
                value = float(element.get("value", current))
                if "tf" in element:
                    part = Hold(tf=float(element["tf"]), value=value)
                else:
                    duration = element.get("duration", element.get("dt"))
                    if duration is None:
                        raise ValueError("constant/hold elements require 'duration' (or 'dt') or 'tf'.")
                    part = Hold(duration=float(duration), value=value)
                parts.append(part)
                current = value
                continue

            if kind == "ramp":
                params = dict(element)
                params.pop("kind", None)
                if not any(k in params for k in ("y0", "y_start")):
                    params["y0"] = current
                part = Ramp(**params)
                parts.append(part)
                current = float(part._resolve(0.0, current).yf)
                continue

            if kind == "harmonic":
                params = dict(element)
                params.pop("kind", None)
                if "y0" not in params and "center" not in params:
                    params["y0"] = current
                part = Harmonic(**params)
                parts.append(part)
                current = float(part._resolve(0.0, current).yf)
                continue

            if kind == "spike":
                # Legacy lab element: spike is two ramp halves with optional cosine shape.
                amp = float(element.get("amplitude", 0.0))
                hp1 = float(element.get("half_period1", 0.0))
                hp2 = float(element.get("half_period2", 0.0))
                if hp1 <= 0.0 or hp2 <= 0.0:
                    raise ValueError("spike elements require half_period1>0 and half_period2>0")
                shape = str(element.get("shape", "cosine")).lower()
                if shape not in {"linear", "cosine"}:
                    raise ValueError("spike shape must be 'linear' or 'cosine'.")

                start_val = current
                peak_val = start_val + amp
                end_val = float(element.get("end_value", start_val))
                if bool(element.get("half_period2_to_start", False)):
                    denom = start_val - peak_val
                    if np.isclose(denom, 0.0):
                        frac = 0.0
                    else:
                        frac = (end_val - peak_val) / denom
                    hp2 = max(np.finfo(float).eps, hp2 * float(np.clip(frac, 0.0, 1.0)))

                up = Ramp(duration=hp1, y0=start_val, yf=peak_val, shape=shape)
                down = Ramp(duration=hp2, y0=peak_val, yf=end_val, shape=shape)
                parts.extend([up, down])
                current = end_val
                continue

            raise ValueError(f"Unknown forcing element kind '{kind}'.")

        sequence = ForcingSequence(parts=parts, label=label)
        return cls(data=sequence)

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
        """Create a forcing object from packaged datasets or an arbitrary CSV file."""

        if dataset is not None:
            if dataset == "vieira_tsi":
                my_resources = importlib.resources.files("paleobeasts") / "data"
                file_path = my_resources.joinpath("vieira_tsi.csv")
                default_time = "Age (yrs BP)"
                default_value = "0"
            elif dataset == "insolation":
                my_resources = importlib.resources.files("paleobeasts") / "data"
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
        """Get forcing value(s) at time ``t``."""
        return self.forcing_func(t)


_VALID_ATTACHMENT_STYLES = {"replacement", "additive"}
_VALID_TIMINGS = {"pre", "post"}


@dataclass
class ForcingSpec:
    """Registration record for a single forcing attachment on a model variable.

    Created internally by ``PBModel.set_forcing``; users do not construct this
    directly.  All fields are validated at construction time.

    Parameters
    ----------
    forcing_object :
        A ``Forcing`` instance, a callable ``f(t)`` → scalar/array, or any
        object with a ``get_forcing(t)`` method.
    attachment_style : {"replacement", "additive"}
        How the forcing value is applied to the target variable.
        ``"replacement"`` substitutes the current parameter or state value.
        ``"additive"`` adds the forcing value to the existing value or to
        ``dx/dt`` for state variables.
    timing : {"pre", "post"}
        When the forcing is applied relative to the integration step.
        ``"pre"`` — applied inside the RHS before the integrator advances.
        ``"post"`` — applied as a correction after the integrator step.

    Notes
    -----
    Derivation rules (enforced by ``set_forcing``, not here):

    * parameter + replacement  →  timing always ``"pre"``
    * state + replacement      →  timing always ``"post"``
    * state + additive         →  timing required from caller
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
        """Return the forcing value at time ``t``."""
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
