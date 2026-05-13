# PaleoBeasts Core Module Reference

> **Maintenance note:** Update this document when functionality in `paleobeasts/core/` is added, changed, or removed.

---

## Overview

`paleobeasts/core` is the foundational layer for all paleoclimate signal models in PaleoBeasts. Its purpose is to provide minimal model "organisms" for studying key aspects of climate dynamics — chaos, multiple equilibria, tipping points — and how those signals are recorded in paleoclimate archives (ice cores, sediment cores, tree rings, etc.).

The module exports two public objects:

| Export | Source | Role |
|--------|--------|------|
| `PBModel` | `pbmodel.py` | Parent class for all concrete models |
| `Forcing` | `forcing.py` | Unified time-varying parameter wrapper |

Concrete models subclass `PBModel` and are housed in `paleobeasts/models/`.

---

## PBModel (`pbmodel.py`)

`PBModel` is an abstract parent class. It is not instantiated directly; child classes override `dydt()` to define their ODE(s) and optionally override hook methods for diagnostics and stochastic noise.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `variable_name` | `str` | Human-readable name for the model's primary output variable |
| `forcing` | `Forcing` | Primary time-varying forcing parameter |
| `state_variables_names` | `list[str]` | All state variable names (integrated + non-integrated) |
| `integrated_state_vars` | `list[str]` | Variables that are actual ODE unknowns |
| `non_integrated_state_vars` | `list[str]` | Variables computed from integrated state (diagnostic states) |
| `diagnostic_variables` | `dict[str, list]` | Named diagnostic outputs accumulated during integration |
| `param_values` | `dict` | Maps parameter names → constants, callables, or Forcing objects |
| `parameter_contract` | `str` | `'legacy'` or `'strict'` — controls callable parameter dispatch |
| `state_variables` | structured ndarray | Named array holding resolved state post-integration |
| `solution` | scipy/custom solution | Raw solver output |
| `time` | `ndarray` | Time axis of the solution |
| `t_span`, `y0`, `t_eval` | array-like | Integration configuration |
| `method` | `str` | Integration method: `'RK45'`, `'euler'`, `'euler_maruyama'` |
| `run_name` | `str` | Human-readable label for a run |
| `rng` | `numpy.random.Generator` | RNG for stochastic methods |

---

### Methods

#### Parameter Resolution

| Method | Signature | Description |
|--------|-----------|-------------|
| `resolve_param` | `(param, t, state) → value` | Dispatches constants, callables, or Forcing objects to their evaluation |
| `get_param` | `(name, t, state) → value` | Looks up `param_values[name]` and resolves it |
| `set_param` | `(name, value)` | Updates `param_values` and keeps instance attributes in sync |
| `_call_param` | `(param, t, state)` | Legacy-mode dispatch; inspects signature heuristically; issues deprecation warnings for non-strict signatures |
| `_call_param_strict` | `(param, t, state)` | Strict-mode dispatch; enforces signatures `(t)`, `(t, state)`, or `(t, state, model)` only |

**Parameter contract modes:**

| Contract | Behavior |
|----------|----------|
| `'legacy'` | Flexible heuristic routing; warns on non-strict signatures |
| `'strict'` | Enforces `(t)`, `(t, state)`, or `(t, state, model)`; raises `TypeError` otherwise |

The `'strict'` mode exists to enable gradual migration away from legacy callable conventions.

---

#### Integration

| Method | Signature | Description |
|--------|-----------|-------------|
| `integrate` | `(t_span, y0, method='RK45', kwargs=None, run_name=None)` | Main entry point; validates state, routes to solver, calls `post_integrate` |
| `dydt` | `() → ndarray` | The ODE right-hand side. **Must be overridden by child classes.** |
| `validate_initial_state` | `(y0) → ndarray` | Checks y0 dimensions against `integrated_state_vars`; returns flat float array |

Supported methods:
- `'RK45'` and other scipy methods → dispatched to `scipy.integrate.solve_ivp`
- `'euler'` → custom fixed-step Euler solver (`paleobeasts.utils.solver`)
- `'euler_maruyama'` → stochastic Euler-Maruyama; requires child class to implement `sde_noise()`

Custom time step for Euler methods: pass `kwargs={'dt': value}`.

---

#### State Management

| Method | Signature | Description |
|--------|-----------|-------------|
| `build_state_from_history` | `(time, history) → structured array or ndarray` | Converts raw solver output into a numpy structured array with named fields (when `state_variables_names` is set) |
| `post_integrate` | `(time, history)` | Post-solve hook: calls `build_state_from_history`, `populate_diagnostics_from_history`, `finalize_diagnostics` |
| `reframe_time_axis` | `(t_eval, update_state=True) → structured array` | Interpolates solution onto a new time grid using dense output (RK45) or linear interpolation (Euler fallback) |

Named structured arrays allow semantic access: `state['temperature']` instead of `state[:, 3]`.

---

#### Diagnostics

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_series_by_name` | `(var_name) → (values, location)` | Returns `(ndarray, "state"\|"diagnostic")` for any named variable |
| `populate_diagnostics_from_history` | `(time, history)` | Hook for child classes to fill `diagnostic_variables`; called post-integration when `uses_post_history()` is True |
| `finalize_diagnostics` | `()` | Converts all diagnostic lists to numpy arrays |
| `uses_post_history` | `() → bool` | Override to return `True` if the model needs `populate_diagnostics_from_history` called |

---

#### Noise Overlays

Non-destructive noise system for simulating paleoclimate archive degradation (taphonomy).

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_noise` | `(var_name, noise_ts)` | Adds noise array to a state or diagnostic variable; preserves original for restoration |
| `remove_noise` | `(var_name)` | Restores pre-noise values for a variable |
| `_reset_noise_overlays` | `()` | Clears all noise tracking at the start of a new integration run |

---

#### Function Replacement

| Method | Signature | Description |
|--------|-----------|-------------|
| `set_function` | `(name, function, bind=None)` | Replaces an instance method at runtime; auto-detects binding via first parameter name inspection |

---

#### Export

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_pyleo` | `(var_names) → pyleoclim.Series or pyleoclim.MultipleSeries` | Exports one or more solution variables to PyLEOclim for downstream timeseries analysis |

---

## Forcing (`forcing.py`)

`Forcing` wraps any time-varying parameter source — analytical sequences, interpolated arrays, or arbitrary callables — behind a single `get_forcing(t)` interface. This is what `PBModel.resolve_param` calls when a parameter is a `Forcing` object.

### ForcingElement Hierarchy

All segments are subclasses of `ForcingElement`. Segments compose via `+` to form `ForcingSequence`.

#### `Hold` — constant segment

| Parameter | Description |
|-----------|-------------|
| `value` | Constant output value |
| `duration` / `dt` | Length of segment (relative time) |
| `tf` | Absolute end time (alternative to duration) |

#### `Ramp` — linear or cosine transition

| Parameter | Description |
|-----------|-------------|
| `y0` / `y_start` | Starting value (inferred from previous segment if omitted) |
| `yf` / `y_end` | Ending value |
| `A` | Amplitude (alternative: `yf = y0 + A`) |
| `y_exit` | Exit value with automatic duration via `duration = half_period * (yf - y0) / A` |
| `half_period` | Used with `y_exit` for duration scaling |
| `shape` | `"linear"` (default) or `"cosine"` (easing) |

#### `Harmonic` — sinusoidal segment

| Parameter | Description |
|-----------|-------------|
| `duration` | Length of segment |
| `period` | Oscillation period |
| `A` | Amplitude |
| `center` | Center value (inferred from `y0` if omitted) |
| `y0` | Starting phase point |

Phase offset `phi` is computed automatically so the sinusoid starts at `y0`.

---

### `ResolvedSegment` (dataclass, frozen)

Immutable compiled representation of a segment:

```python
@dataclass(frozen=True)
class ResolvedSegment:
    kind: str        # "hold", "ramp", "harmonic"
    t0, tf: float    # time bounds
    y0, yf: float    # value bounds
    eval_mode: str   # "constant", "linear", "cosine", "harmonic"
    params: dict     # mode-specific (e.g. {"value": ...}, {"center": ..., "A": ...})
```

---

### `ForcingSequence`

An ordered sequence of `ForcingElement` objects with time/value continuity.

| Method | Description |
|--------|-------------|
| `compile()` | Resolves all segments; returns dict with `"segments"`, `"t_end"`, `"y_start"`, `"y_end"`, `"transition_times"`, `"n_transitions"` |
| `__call__(t)` | Evaluates forcing at scalar or array `t` |
| `summary()` | Returns high-level summary of compiled sequence |
| `__add__(other)` | Compose sequences: `seq1 + seq2` or `seq1 + element` |

---

### `Forcing` — unified interface

```python
Forcing(data, time=None, params=None, interpolation="cubic")
```

`data` can be:
1. A `ForcingElement` or `ForcingSequence` — event-based analytical forcing
2. A callable — arbitrary function `f(t)`
3. An array-like — interpolated time series

| Parameter | Description |
|-----------|-------------|
| `time` | Time axis for array data (defaults to `np.arange(len(data))`) |
| `interpolation` | `"cubic"` (CubicSpline, extrapolates) or `"linear"` (interp1d, clamps at edges) |

**Public method:** `get_forcing(t)` — evaluates forcing at time(s) `t`.

#### Constructors

| Constructor | Description |
|-------------|-------------|
| `Forcing(data, ...)` | Direct construction from element, callable, or array |
| `Forcing.from_sequence(parts, label)` | From an iterable of `ForcingElement` objects |
| `Forcing.from_elements(elements, y0, label)` | Dictionary-based syntax (see below) |
| `Forcing.from_csv(dataset, file_path, value_name, time_name, ...)` | Load from CSV or packaged datasets |

**`from_elements` dict syntax:**

```python
{"kind": "hold",     "value": X, "duration": Y}
{"kind": "ramp",     "y0": ..., "yf": ..., "duration": ...}
{"kind": "harmonic", "duration": ..., "period": ..., "A": ...}
{"kind": "spike",    "amplitude": ..., "half_period1": ..., "half_period2": ...}  # legacy
```

Value continuity is auto-chained across elements.

**Packaged datasets for `from_csv`:**

| Dataset key | Contents |
|-------------|----------|
| `"vieira_tsi"` | Total solar irradiance reconstruction (Vieira et al.) |
| `"insolation"` | 65°N summer insolation |

---

## Design Patterns

### Parameter dispatch
`resolve_param` handles three types uniformly: constants are returned as-is, callables are routed via signature inspection (legacy) or contract enforcement (strict), and `Forcing` objects are called via `get_forcing(t)`. Child classes never need to handle dispatch themselves.

### Named structured arrays for state
When `state_variables_names` is provided, `build_state_from_history` wraps the raw solver output in a numpy structured array. This enables `state['temperature']` instead of `state[:, 3]`.

### Post-integration hooks
`uses_post_history()` and `populate_diagnostics_from_history()` let child classes compute derived quantities from the full integration history without polluting `dydt`.

### Composable forcing elements
The `+` operator on `ForcingElement` / `ForcingSequence` enables declarative construction of complex forcings:

```python
forcing = Forcing(
    Hold(value=1361, duration=500) +
    Ramp(yf=1358, duration=200) +
    Harmonic(duration=10000, period=1000, A=2)
)
```

### Noise overlays (non-destructive)
`add_noise` / `remove_noise` preserve the original signal, enabling repeated experiments with different noise realizations on a single model run without re-integrating.

---

## Key Domain Concepts

| Term | Definition |
|------|-----------|
| **Paleoclimate archive** | Physical record of past climate (ice core, sediment core, tree ring, speleothem) |
| **Forcing** | External driver imposed on the climate system (solar irradiance, CO₂, volcanic aerosols) |
| **State variable** | Quantity that evolves according to an ODE (e.g., temperature, ice volume) |
| **Integrated state var** | State variable that is directly integrated (appears in `dydt`) |
| **Non-integrated state var** | Quantity computed from integrated state; not an ODE unknown |
| **Taphonomy** | Physical/chemical alteration of a record after deposition (causes signal degradation) |
| **Diagnostic variable** | Derived quantity recorded during integration but not integrated (e.g., effective albedo) |

---

## Ecosystem Integration

| System | How it connects |
|--------|----------------|
| **PyLEOclim** | `to_pyleo(var_names)` exports any named variable as `pyleoclim.Series` or `MultipleSeries` for spectral analysis, tipping point detection, etc. |
| **scipy** | `integrate` dispatches to `scipy.integrate.solve_ivp` for `'RK45'` and other adaptive methods |
| **Custom solvers** | `paleobeasts.utils.solver` provides fixed-step Euler and Euler-Maruyama (stochastic) solvers |
