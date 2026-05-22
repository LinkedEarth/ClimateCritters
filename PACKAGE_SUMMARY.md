# PaleoBeasts — Package Summary

## Package Structure

```
paleobeasts/
├── core/                   — Base class and forcing framework
├── signal_models/          — Physical/climate ODE models
├── utils/                  — Solvers, noise, plotting, resampling
├── data/                   — Data directory (placeholder)
├── reference_content/      — Interactive paleoclimate event labeling
└── tests/                  — Test suite
```

---

## Core Framework (`core/`)

### `PBModel` ([paleobeasts/core/pbmodel.py](paleobeasts/core/pbmodel.py))

The abstract base class for all signal models. Subclasses must implement `dydt(t, x)`. Key capabilities:

- **Flexible parameter system** — any parameter can be a constant, a callable with signature `(t)`, `(t, state)`, `(t, state, model)`, or an object with a `get_forcing()` method. The base class inspects the signature and dispatches accordingly.
- **Integration** — wraps `scipy.integrate.solve_ivp` (default RK45) and provides hand-rolled Euler, Euler-Maruyama, and fixed-step RK4 solvers via `integrate(t_span, y0, method=...)`.
- **Post-integration hook** — subclasses that need the full trajectory (e.g. to compute diagnostics) override `uses_post_history() → True` and populate state/diagnostics in `post_integrate()`. Models that do not use this hook incorrectly append to `self.state_variables` inside `dydt`, which runs at every sub-step.
- **Runtime customization** — `set_param()` and `set_function()` allow modifying individual parameters or replacing methods on a specific instance without touching the class definition.
- **Pyleoclim integration** — `to_pyleo()` converts any state or diagnostic variable to a `pyleoclim.Series`.
- **Noise and resampling** — `add_noise()` / `remove_noise()` layer externally generated noise onto a variable; `reframe_time_axis()` resamples the solution to a new time grid.

### `Forcing` and Forcing Elements ([paleobeasts/core/forcing.py](paleobeasts/core/forcing.py))

A unified forcing wrapper that accepts callables, time-series arrays, or composable `ForcingElement` sequences. Three element types are provided:

| Class | Description |
|---|---|
| `Hold` | Constant segment over a specified duration or until an absolute time |
| `Ramp` | Monotonic transition with linear or cosine easing |
| `Harmonic` | Sinusoidal segment constrained by duration, period, and amplitude |

`ForcingSequence` chains elements together; `Forcing.from_sequence()` and `Forcing.from_elements()` provide builder-pattern constructors. `Forcing.from_csv()` loads packaged datasets (e.g. orbital insolation) or arbitrary CSV files.

---

## Signal Models (`signal_models/`)

### Climate & Earth System Models

| Model | Class | File | State Variables | Key Physics |
|---|---|---|---|---|
| Ganopolski (2024) | `Model3` | [g24.py](paleobeasts/signal_models/g24.py) | `v`, `k` | Ice volume with orbital forcing and glacial–interglacial bifurcation |
| Energy Balance | `EBM` | [ebm.py](paleobeasts/signal_models/ebm.py) | `T` | 0D radiative balance (albedo, OLR, solar constant) |
| Latitudinal EBM | `LatitudinalEBM` | [latitudinal_ebm.py](paleobeasts/signal_models/latitudinal_ebm.py) | `T_0…T_n` | 1D diffusive EBM on a latitude grid with ice-line feedback |
| Two-Box Carbon | `TwoBoxCarbon` | [two_box_carbon.py](paleobeasts/signal_models/two_box_carbon.py) | `A`, `S` | Atmosphere–surface-ocean carbon exchange with volume-aware fluxes |
| Generic Box Model | `GenericBoxModel` via `BoxModelSpec` | [box_model.py](paleobeasts/signal_models/box_model.py) | configurable | Declarative multi-box ODE builder; supports reciprocal exchange and directed transport |
| ENSO Recharge | `ENSORechargeOscillator` | [enso_recharge.py](paleobeasts/signal_models/enso_recharge.py) | `T`, `h` | Jin-style ENSO recharge oscillator with seasonal forcing |
| Stommel THC | `Stommel` | [stommel.py](paleobeasts/signal_models/stommel.py) | `T`, `S` | 2-box thermohaline circulation; diagnostic: overturning strength `q` |
| Daisyworld | `Daisyworld` | [daisyworld.py](paleobeasts/signal_models/daisyworld.py) | `Aw`, `Ab`, `T` | Biosphere–climate feedback through daisy-coverage-mediated albedo |
| Bipolar Seesaw (minimal) | `Stocker2003BipolarSeesaw` | [stocker2003_bipolar_seesaw.py](paleobeasts/signal_models/stocker2003_bipolar_seesaw.py) | `Ts` | Single-variable thermal relaxation to prescribed northern forcing |
| Bipolar Seesaw (extended) | `Stocker2003ExtendedSeaIceSeesaw` | [stocker2003_bipolar_seesaw.py](paleobeasts/signal_models/stocker2003_bipolar_seesaw.py) | `T_R`, `T_S`, `A`, `T_ANT` | Four-variable system with reservoir, Southern Ocean, sea-ice, and Antarctic temperatures |

### Dynamical Systems & Oscillators

| Model | Class(es) | File | State Variables | Key Physics |
|---|---|---|---|---|
| Lorenz (1963) | `Lorenz63` | [lorenz.py](paleobeasts/signal_models/lorenz.py) | `x`, `y`, `z` | Classic 3-variable chaos; butterfly attractor |
| Lorenz (1996) | `Lorenz96` | [lorenz.py](paleobeasts/signal_models/lorenz.py) | `x_k` (+ `y_jk` for two-scale) | Atmospheric chaos on a ring; supports fast–slow timescale separation |
| Roessler | `Roessler` | [roessler.py](paleobeasts/signal_models/roessler.py) | `x`, `y`, `z` | Spiral chaotic oscillator (simpler than Lorenz) |
| Damped Spring | `DampedSpring` | [damped_spring.py](paleobeasts/signal_models/damped_spring.py) | `x`, `v` | Damped/driven spring-mass system; diagnostics: energy, ω₀ |
| Simple Pendulum | `SimplePendulum` | [pendulum.py](paleobeasts/signal_models/pendulum.py) | `θ`, `ω` | Nonlinear pendulum with optional linear damping |
| Driven Pendulum | `DrivenPendulum` | [pendulum.py](paleobeasts/signal_models/pendulum.py) | `θ`, `ω` | Forced damped pendulum; canonical 1D chaos testbed |
| Double Pendulum | `DoublePendulum` | [pendulum.py](paleobeasts/signal_models/pendulum.py) | `θ₁`, `ω₁`, `θ₂`, `ω₂` | Conservative chaotic system; diagnostics: energy, Cartesian positions |

---

## Utilities (`utils/`)

| Module | Key Functions | Purpose |
|---|---|---|
| [solver.py](paleobeasts/utils/solver.py) | `euler_method`, `euler_maruyama_method`, `flux_divergence`, `define_t_eval` | Integration backends and finite-volume helpers |
| [forcing_utils.py](paleobeasts/utils/forcing_utils.py) | `create_sinusoid_forcing`, `create_periodic_forcing`, `create_constant_forcing`, `create_piecewise_forcing` | Convenience constructors for common `Forcing` shapes |
| [noise.py](paleobeasts/utils/noise.py) | `from_series`, `from_param` | AR(1), colored, and fractional Gaussian noise via pyleoclim |
| [resample.py](paleobeasts/utils/resample.py) | `downsample` | Non-uniform time axis subsampling (exponential, Poisson, Pareto, random) |
| [plotting_utils.py](paleobeasts/utils/plotting_utils.py) | `plot_solvers` | Side-by-side solver trajectory comparison |
| [func.py](paleobeasts/utils/func.py) | `smooth_and_interpolate`, `make_derivative_func` | Moving-average smoothing, CubicSpline derivative estimation |
| [constants.py](paleobeasts/utils/constants.py) | `sigma` | Physical constants (Stefan-Boltzmann) |

---

## Notebooks

Notebooks live in two subdirectories of [`notebooks/`](notebooks/):

### Model Demos (`notebooks/model_demos/`)

**[Ganopolski2024_demo.ipynb](notebooks/model_demos/Ganopolski2024_demo.ipynb)**
Demonstrates `Model3` (ice volume with orbital forcing). Runs the model under both synthetic periodic forcing and real orbital insolation data loaded via `Forcing.from_csv()`. Includes a parameter sweep over timestep sizes (dt = 2–12 kyr) illustrating sensitivity to step size and verifying convergence within 200 kyr as described in the paper.

**[Ganopolski2024_strict_contract_validation.ipynb](notebooks/model_demos/Ganopolski2024_strict_contract_validation.ipynb)**
Advanced `Model3` usage showing strict parameter contract validation and runtime function injection. Demonstrates how to override the regime-transition logic (`calc_k`) on a single instance using `set_function()` without modifying the class, and compares the resulting trajectories side-by-side.

**[daisyworld_demo.ipynb](scratchwork/notebooks/daisyworld_demo.ipynb)**
Minimal introduction to the Daisyworld self-regulating climate model. Shows unforced equilibrium behavior alongside a periodically forced variant (sinusoidal luminosity), illustrating how planetary albedo and temperature stabilize through daisy-coverage feedback.

**[ebm_demo.ipynb](notebooks/model_demos/ebm_demo.ipynb)**
Explores the 0D Energy Balance Model across three forcing configurations: an analytical sinusoidal function, a sampled data array with cubic interpolation, and observational Total Solar Irradiance data. Demonstrates time-varying parameters (heat capacity `C(t)`, albedo `α(T)`) and compares RK45 vs. Euler integration over a 10,000 kyr run.

**[lorenz63.ipynb](notebooks/model_demos/lorenz63.ipynb)**
Comprehensive treatment of the Lorenz-63 system. Reproduces canonical figures from the 1963 paper, visualizes the butterfly attractor (x-z and y-z projections), demonstrates sensitive dependence on initial conditions with an ε = 10⁻⁸ perturbation, and explores sinusoidal forcing and time-varying parameters that cross the chaos onset boundary (ρ ≈ 24.74).

**[lorenz96.ipynb](notebooks/model_demos/lorenz96.ipynb)**
In-depth walkthrough of Lorenz-96 in both single-scale (n=40, F=8) and two-scale (K=36 slow, J=10 fast) configurations. Covers periodic forcing, stochastic forcing via Euler-Maruyama, warmup procedures for reaching the attractor, and fixed-step RK4 requirements for the two-scale system. Space-time heatmaps illustrate westward-propagating wave packets and fast-variable envelope modulation.

**[stommel_demo.ipynb](scratchwork/in_progress/stommel_demo.ipynb)**
Brief demonstration of the Stommel thermohaline circulation model. Compares unforced equilibrium behavior to a sinusoidally perturbed run, with the diagnostic overturning transport `q` plotted alongside the thermal and haline contrasts.

**[two_box_carbon_demo.ipynb](scratchwork/in_progress/two_box_carbon_demo.ipynb)**
Covers both the concrete `TwoBoxCarbon` model and the generic `BoxModelSpec` architecture. Verifies mass conservation (total drift < 10⁻⁶) in the closed system, reconstructs the same model using the declarative spec API, then builds a five-box Atlantic–Pacific–Southern Ocean network combining reciprocal exchange (diffusive mixing) and directed transport (overturning circulation).

---

### Functionality Demos (`notebooks/functionality_demos/`)

**[downsample_demo.ipynb](notebooks/functionality_demos/downsample_demo.ipynb)**
Demonstrates `utils.resample.downsample()` using SOI (Southern Oscillation Index) data. Walks through all four sampling methods — exponential, Poisson, Pareto, and random — with parameter sweeps showing how each method affects the resulting time-axis resolution distribution.

**[model_noise_demo.ipynb](notebooks/functionality_demos/model_noise_demo.ipynb)**
End-to-end workflow combining `Model3` integration, AR(1) noise addition, and downsampling. Represents a realistic pipeline for generating synthetic paleoclimate proxy records: deterministic model output → noise → sparse irregular sampling → CSV export with a common resampled time axis.

**[noise_demo.ipynb](notebooks/functionality_demos/noise_demo.ipynb)**
Reference guide for noise generation via `utils.noise`. Covers both pathways: `from_series` (match the autocorrelation or spectrum of an existing series) and `from_param` (specify distribution type and parameters directly). Methods include AR(1) simulation, phase randomization, colored noise, fractional Gaussian noise, and white noise. Demonstrates ensemble generation and the `plot_envelope` visualization.

**[solver_choice.ipynb](notebooks/functionality_demos/solver_choice.ipynb)**
Pedagogical comparison of `RK45`, `euler`, and `euler_maruyama` for two representative model types: smooth chaos (Lorenz-96) and regime-switching (Model 3). Shows why adaptive solvers corrupt the regime-transition history in `Model3` and provides practical guidance on timestep sizing and convergence checking.

---

## Notes: Missing Docstrings

- **`solver.py`** — `Solution` class has no docstring; `euler_method` and `euler_maruyama_method` lack parameter descriptions.
- **`plotting_utils.py`** — `plot_solvers` has no docstring.
- **`func.py`** — `make_derivative_func` is flagged `UNTESTED` in a code comment but carries no warning in the docstring and is publicly exported. `smooth_and_interpolate` has no parameter table.
- **`taxonomy_entry.py`** — Most interactive functions (`label_data`, `gen_fit`, `save_data`) lack parameter/return docstrings entirely.
- **`box_model.py`** — `BoxModelContext` has no class-level docstring.
- **`Stocker2003ExtendedSeaIceSeesaw`** — The docstring parameter list does not cover all 12+ parameters.

---

## Notes: Redundancy & Design Issues

1. **State-appending in `dydt`** — `Lorenz63`, `Roessler`, `Daisyworld`, `EBM`, and `Model3` all append to `self.state_variables` and `self.time` inside `dydt`, which runs at every integration sub-step. The intended pattern (used by Stommel, TwoBoxCarbon, ENSORechargeOscillator, and others) is `uses_post_history() → True` with state populated in `post_integrate()`. These five models are inconsistent with the rest of the codebase.

2. **Duplicated `_forcing_vector()`** — `Lorenz63` and `Stommel` each implement identical private methods that broadcast a scalar forcing value to a vector. This logic could live in `PBModel`.

3. **Duplicated oscillator helpers** — `SimplePendulum` and `DampedSpring` both define `natural_frequency()`, `natural_period()`, and `damping_ratio()` with identical implementations. A shared mixin for oscillatory models would eliminate the duplication.

4. **`EBM.calc_merid_diff` stub** — The method is registered as a parameter and called in `dydt` but always returns zero. There is an internal TODO comment. The stub should either be removed or replaced with a clear `NotImplementedError` / documented as a deliberate no-op.

5. **`func.make_derivative_func` — untested and undocumented** — Flagged `UNTESTED` in the source but publicly exported with no docstring-level warning. Should either be tested, moved to a private namespace, or emit a `warnings.warn` at call time.
