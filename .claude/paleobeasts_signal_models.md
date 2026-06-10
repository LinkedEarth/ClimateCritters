# PaleoBeasts Signal Models Reference

> **Maintenance note:** Update when models are added, removed, or refactored. Cross-reference with [paleobeasts_core.md](paleobeasts_core.md) when core API changes affect conformance.

---

## Quick-Reference Summary

| Model | Class | File | State vars | Diagnostic vars | Forcing | Stochastic | Notebook(s) | PBModel conformant |
|-------|-------|------|------------|-----------------|---------|------------|-------------|-------------------|
| Lorenz 63 | `Lorenz63` | `lorenz.py` | x, y, z | — | scalar/3-vector additive | No | `lorenz63_demo` | ⚠ Side effects in dydt |
| Energy Balance | `EBM` | `ebm.py` | T | albedo, absorbed_SW, OLR, solar_incoming | S₀(t) | No | `ebm_demo` | ⚠ Side effects in dydt |
| Stommel THC | `Stommel` | `stommel.py` | T, S | q | freshwater/2-vector | No | `stommel_demo` | ⚠ Side effects in dydt |
| Ganopolski 2024 | `Model3` | `g24.py` | v, k | insolation | orbital forcing (required) | No | `Ganopolski2024_demo`, `_strict_contract_validation`, `model_solver_choice_G24` | ⚠ Side effects in dydt |
| Melcher 2025 DO | `Melcher2025DOModel` | `melcher2025_do.py` | delta_b, B | q, amoc_dim, aabw_dim | None | Yes (Euler-Maruyama) | `melcher2025_do_demo` | ✓ |
| Stocker 2003 Bipolar Seesaw | `Stocker2003BipolarSeesaw` | `stocker2003_bipolar_seesaw.py` | Ts | Tn | T_N(t) | No | `stocker2003_bipolar_seesaw_demo` | ✓ |
| Stocker 2003 Extended Sea-Ice | `Stocker2003ExtendedSeaIceSeesaw` | `stocker2003_bipolar_seesaw.py` | T_R, T_S, A, T_ANT | T_N | T_N(t) | No | `stocker2003_extended_seaice_demo` | ✓ |
| Lorenz 96 | `Lorenz96` | `lorenz96.py` | x0…xₙ | — | F(t) | No | `lorenz96_demo` | ⚠ Side effects in dydt |
| Lorenz 96 Two-Scale | `Lorenz96TwoScale` | `lorenz96.py` | x0…x_K, y0…y_KJ | — | F(t) | No | (none dedicated) | ⚠ Side effects in dydt + custom `integrate()` |
| Daisyworld | `Daisyworld` | `daisyworld.py` | Aw, Ab, T | A_planet, A_bare, beta_w, beta_b | L(t) additive | No | `daisyworld_demo` | ⚠ Side effects in dydt |
| Insolation utilities | — | `insolation.py` | N/A | N/A | N/A | N/A | (none) | N/A — not a model class |

---

## Design Conformance

The `PBModel` design expects `dydt` to be a pure function: compute derivatives, return them, nothing else. State reconstruction happens post-integration via `post_integrate` → `build_state_from_history`. Diagnostics are populated post-solve via `populate_diagnostics_from_history` (when `uses_post_history()` returns `True`).

### Conformant pattern (Melcher2025DO, Stocker2003 family)
```
integrate() → solver calls dydt() [pure] → post_integrate() → populate_diagnostics_from_history()
```

### Non-conformant pattern (Lorenz63, EBM, Stommel, G24, Lorenz96, Daisyworld)
```
integrate() → solver calls dydt() [accumulates state + diagnostics as side effect]
```
The non-conformant models manually concatenate to `self.state_variables` and append to `self.time` on every `dydt` call. This is functional but inefficient (O(n²) memory due to repeated concatenation), makes `dydt` non-reentrant, and conflicts with PBModel's post-solve state building.

---

## Model Details

### Lorenz63

**What it models:** The Lorenz (1963) chaotic atmospheric convection system — 3-variable deterministic chaos, sensitive to initial conditions.

**Equations:**
```
dx/dt = σ(y − x) + f[0]
dy/dt = x(ρ − z) − y + f[1]
dz/dt = xy − βz + f[2]
```

**Constructor:**
```python
Lorenz63(forcing, var_name='lorenz63', sigma=10.0, rho=28.0, beta=8/3,
         state_variables=['x','y','z'], diagnostic_variables=[])
```

**Parameters in `param_values`:** `sigma`, `rho`, `beta` — all support callables and `Forcing` objects.

**Forcing:** Scalar → added to `dx/dt` only (broadcast as `[f, 0, 0]`). 3-element array → added component-wise. No forcing → zero vector.

**Additional methods:**
- `_forcing_vector(t)` — converts forcing input to 3-element vector. Necessary; no redundancy.

**Hooks used:** None (`uses_post_history` not overridden).

**Parameter contract:** `'legacy'` (default).

**Conformance issues:**
- Appends to `self.state_variables` and `self.time` inside `dydt` on every solver step. No diagnostics are declared, so no diagnostic side effects.

**Demo — `lorenz63_demo.ipynb`:**
- Unforced butterfly attractor (RK45, reframed to even grid via `reframe_time_axis`)
- Sinusoidal forced attractor showing phase-space distortion
- Time-varying σ and ρ via callables `sigma_func(t, x, model)` and `rho_func(t)` — exercises the parameter dispatch system

---

### EBM (Energy Balance Model)

**What it models:** 0-D planetary energy balance: surface temperature evolving from the balance between absorbed shortwave radiation and outgoing longwave radiation.

**Equations:**
```
dT/dt = (1/C) × (absorbed_SW − OLR + merid_diff)

absorbed_SW = (1 − albedo) × S₀(t) / 4
OLR = σ × (T × pRad/ps)^(2/7) × T⁴  [default Stefan-Boltzmann with pressure ratio]
```

**Constructor:**
```python
EBM(forcing, state_variables=['T'],
    diagnostic_variables=['albedo','absorbed_SW','OLR','solar_incoming'],
    var_name='temperature', OLR=None, C=4, merid_diff=0, albedo=0.3)
```

**Parameters in `param_values`:** `C`, `albedo`, `OLR`, `merid_diff` — all support callables (state-dependent) and `Forcing` objects.

**Forcing:** Provides S₀(t); accessed via `forcing.get_forcing(t)`. Required.

**Helper functions (module-level):**
- `albedo_func(ebm_model, Ts, alpha_ice, alpha_0, T1, T2)` — temperature-dependent ice-albedo feedback (parabolic transition). Primary way to add ice-albedo feedback.
- `OLR_func(pRad, ps)` — factory returning the default Stefan-Boltzmann OLR callable.
- `albedo_func1D(...)` — meridional (1-D) stub; incomplete (references `ebm_model.phi`).
- `incoming_SW_func(t, S_0)` and `calc_f(t, S_0, T1)` — placeholder / periodic forcing helpers.

**Additional methods (instance):**
- `calc_albedo(T, t)`, `calc_OLR(T, t)`, `calc_merid_diff(T, t)`, `calc_C(T, t)` — thin wrappers around `get_param()`. Provide a named override surface for subclasses but add no logic of their own; could be removed in a cleanup.

**Hooks used:** None (`uses_post_history` not overridden, though `to_pyleo()` is used in the demo to convert diagnostics).

**Parameter contract:** `'legacy'` (default).

**Conformance issues:**
- Appends state, diagnostics, and time inside `dydt` on every solver step.
- Diagnostic appending is aggressive — runs at every function evaluation, not just solution points (adaptive solvers evaluate `dydt` many more times than there are output points).
- 1-D meridional diffusion (`merid_diff`) is documented as incomplete (TODO in source).

**Demo — `ebm_demo.ipynb`:**
- Superposition forcing (3 sinusoids at 11, 90, 200-year periods)
- Time-varying `C` and `albedo` via `C_func(t, T, model)` / `albedo_func(model, T)`
- Array forcing with linear interpolation vs. smooth function
- Real TSI data via `Forcing.from_csv(dataset='vieira_tsi')`; comparison of RK45 vs. Euler (dt=4) vs. Euler (dt=2)
- Diagnostic stacking with `to_pyleo()` and `stackplot()`

---

### Stommel

**What it models:** 2-box thermohaline circulation — temperature contrast (ΔT) and salinity contrast (ΔS) between equatorial and polar boxes, with bistable overturning strength q.

**Equations:**
```
q = k(αT − βS)
dT/dt = −λ_T(T − T*) − |q|T + f[0]
dS/dt = E − λ_S(S − S*) − |q|S + f[1]
```
The absolute value `|q|` is the bistability-enabling nonlinearity.

**Constructor:**
```python
Stommel(forcing=None, var_name='stommel',
        alpha=1.0, beta=1.0, k=1.0, E=0.0,
        lambda_T=1.0, lambda_S=1.0, T_star=1.0, S_star=0.0,
        state_variables=['T','S'], diagnostic_variables=['q'])
```

**Parameters in `param_values`:** All 8 (`alpha`, `beta`, `k`, `E`, `lambda_T`, `lambda_S`, `T_star`, `S_star`).

**Forcing:** Optional. Scalar → freshwater perturbation added to `dS/dt`. 2-vector → added component-wise to `(dT/dt, dS/dt)`.

**Additional methods:**
- `overturning(t, x)` — computes q from current state. Clean, reusable abstraction.
- `_forcing_vector(t)` — scalar-to-2-vector broadcast, same pattern as Lorenz63.

**Hooks used:** None.

**Parameter contract:** `'legacy'` (default).

**Conformance issues:**
- Appends state (`T`, `S`), diagnostic `q`, and time inside `dydt`.

**Demo — `stommel_demo.ipynb`:**
- Unforced vs. freshwater-forced (period 10, amplitude 0.15) side-by-side
- Three-panel plot of T, S, q time series
- Does not demonstrate bifurcation / bistable regime or parameter time-variation

---

### Ganopolski 2024 — `Model3`

**What it models:** Ice-volume evolution under orbital forcing, with a discrete regime variable k (1 = glacial, 2 = deglaciation) and hysteresis-based threshold transitions. Based on Ganopolski et al. (2024).

**Equations:**
```
If k=1 (glacial):   dv/dt = (ve(v,f) − v) / t1
If k=2 (deglacial): dv/dt = −vc / t2

ve selected from glacial equilibrium vg or unstable vu depending on current v:
vg = 1 + √((f2 − f) / (f2 − f1))
vu = 1 − √((f2 − f) / (f2 − f1))

Regime transitions (calc_k):
  1→2: when df/dt > 0, f > 0, v > vc
  2→1: when f < f1
```

**Constructor:**
```python
Model3(forcing,  # required — orbital insolation
       var_name='ice volume',
       f1=-16, f2=16,       # forcing thresholds (W/m²)
       t1=30, t2=10,        # relaxation timescales (kyr)
       vc=1.4,              # critical ice volume
       state_variables=['v','k'],
       non_integrated_state_vars=['k'],
       diagnostic_variables=['insolation'],
       parameter_contract='legacy')
```

**Parameters in `param_values`:** `f1`, `f2`, `t1`, `t2`, `vc`, `dfdt` (callable for forcing derivative). All support callables and `Forcing` objects.

**Forcing:** Required. Provides insolation f(t). Accessed via `forcing.get_forcing(self.time_util(t))`.

**Additional methods:**
- `calc_k(k, dfdt, f, v, vc, f1)` — regime transition logic. Overridable via `set_function('calc_k', fn)`.
- `calc_ve(v, f, f1, f2)` — equilibrium ice volume (dispatches to `calc_vg` or `calc_vu`).
- `calc_vg(f, f1, f2)` — glacial equilibrium branch.
- `calc_vu(f, f1, f2)` — unstable equilibrium branch.
- `calc_vc(t, x)` — fetch time/state-varying vc.
- `calc_dfdt(t, x)` — forcing derivative; tries callable first, then `resolve_param`.

**Hooks used:** `set_function('calc_k', ...)` demonstrated in strict-contract notebook.

**Parameter contract:** Supports both; `'strict'` validated in `Ganopolski2024_strict_contract_validation.ipynb`.

**Conformance issues:**
- Appends v, k, and diagnostic `insolation` inside `dydt`.
- `k` is non-integrated but maintained by hand in `dydt`, reading from `self.state_variables['k'][-1]` — creates a dependency between consecutive `dydt` calls, which is fine for Euler but fragile for adaptive solvers that may evaluate `dydt` at out-of-order times.

**Demo — `Ganopolski2024_demo.ipynb`:**
- Euler integration with dt sensitivity (dt = 2, 4, 6.5, 8, 12 kyr)
- Real insolation via `Forcing.from_csv(dataset='insolation')` with `time_util` lambda
- Plots v and k (shaded deglaciation phases)

**Demo — `Ganopolski2024_strict_contract_validation.ipynb`:**
- `parameter_contract='strict'`, `set_function('calc_k', custom_fn)` to swap regime logic at runtime
- Side-by-side comparison of default vs. custom regime logic under identical forcing

**Demo — `model_solver_choice_G24.ipynb`:**
- Compares Euler, RK45, LSODA on the model (uses older standalone function style, not the class)

---

### Melcher 2025 — DO Events

**What it models:** Stochastic slow-fast system for Dansgaard-Oeschger events — dimensionless salinity/buoyancy anomaly (delta_b) and overturning feedback (B) with noise-driven transitions between stadial (weak AMOC) and interstadial (strong AMOC) states.

**Equations:**
```
d(delta_b)/dt = −B − |q|(delta_b − b0) + σ·dW
dB/dt = (delta_b + αB − γ) / τ + σ·dW
q = q0 + q1(delta_b − b0)

amoc_dim = ψ0 + ψ1 · delta_b
aabw_dim = ψ_a + χ_a · (b_c / B_c) · B
```

**Constructor:**
```python
Melcher2025DOModel(forcing=None, var_name='melcher2025_do',
    q0=-9.0, q1=12.0, b0=0.625,
    tau=0.902, alpha=-0.6, gamma=1.2, sigma=0.2,
    psi0=-4.5e6, psi1=20.0e6, psi_a=5.0e6, chi_a=2.5,
    b_c=0.004, B_c=3.8e-10,
    state_variables=['delta_b','B'],
    diagnostic_variables=['q','amoc_dim','aabw_dim'])
```

**Parameters in `param_values`:** All 13 parameters. `sigma` controls noise amplitude.

**Forcing:** Not used; model is autonomous.

**Additional methods:**
- `transport(t, x)` — computes q from current state. Clean, reusable.
- `sde_noise(t, x)` — returns `[sigma, sigma]`; required by Euler-Maruyama solver.
- `_redimensionalized_diagnostics(t, x)` — converts dimensionless state to physical AMOC/AABW units.
- `populate_diagnostics_from_history(time, history)` — post-solve hook; recomputes q, amoc_dim, aabw_dim from the full trajectory.

**Hooks used:** `uses_post_history()` returns `True`; `populate_diagnostics_from_history` overridden; `sde_noise` overridden. This is the canonical conformant stochastic model pattern.

**Parameter contract:** `'legacy'` (default); `'strict'` also valid.

**Conformance:** Fully conformant. No side effects in `dydt`.

**Demo — `melcher2025_do_demo.ipynb`:**
- Baseline Euler-Maruyama run (dt=0.01, t=40, seed=42); plots delta_b, B, q
- Parameter sensitivity: varies alpha, gamma, sigma independently
- Time-varying parameters via lambdas
- Reproducibility check (same seed → same trajectory)
- Optional paper-artifact validation: ensemble statistics (E = transition frequency, P = stadial fraction) vs. NGRIP data

---

### Stocker 2003 — Bipolar Seesaw

**What it models:** Minimal 1-ODE thermal seesaw: Southern Hemisphere temperature (Ts) responds to prescribed Northern Hemisphere forcing (Tn) with a single timescale τ.

**Equation:**
```
dTs/dt = (β·Tn(t) − Ts) / τ
```

**Constructor:**
```python
Stocker2003BipolarSeesaw(forcing=None, var_name='stocker2003_bipolar_seesaw',
    tau=1000.0, beta=-1.0, Tn=0.0,
    state_variables=['Ts'], diagnostic_variables=['Tn'])
```

**Parameters in `param_values`:** `tau`, `beta`, `Tn`.

**Forcing:** Provides Tn(t). If not provided, `Tn` constant from `param_values` is used. Validation: raises `ValueError` if `tau <= 0`.

**Additional methods:** None beyond PBModel.

**Hooks used:**
- `uses_post_history()` returns `True`
- `populate_diagnostics_from_history` reconstructs Tn from forcing/param at each timestep
- `to_pyleo()` used in demo; `add_noise()` / `remove_noise()` demonstrated

**Parameter contract:** `'legacy'` (default).

**Conformance:** Fully conformant. Cleanest model in the codebase.

**Demo — `stocker2003_bipolar_seesaw_demo.ipynb`:**
- Periodic on/off north signal with sinusoidal modulation → `Forcing(data=north, time=time, interpolation='linear')`
- AR(1) and white noise workflows via `add_noise()` / `remove_noise()`; restoration verified
- Tau estimation via correlation scan over a grid of tau values
- Optional: pyleoclim Butterworth filtering before analysis

---

### Stocker 2003 — Extended Sea-Ice Seesaw

**What it models:** 4-ODE extension of the bipolar seesaw coupling an ocean reservoir (T_R), Southern Ocean (T_S), sea-ice area fraction (A), and Antarctic surface temperature (T_ANT). Captures sea-ice feedback and reservoir dynamics.

**Equations:**
```
dT_R/dt   = (−(T_R − T_N) + ε_R) / τ_R
dT_S/dt   = (κ(T_R − T_S) − λ_S(T_S − T_S0) + α(1 − A_eff) + ε_S) / τ_S
dA/dt     = (−β(T_S − T_S0) − γ·A_eff(1 − A_eff)(T_S − T_c) + ε_A) / τ_A
dT_ANT/dt = (δ(T_S − T_ANT) + η(1 − A_eff) + ε_ANT) / τ_ANT

A_eff = clip(A, 0, 1)    [boundary derivative suppression at A=0 and A=1]
```

**Constructor:**
```python
Stocker2003ExtendedSeaIceSeesaw(forcing=None,
    var_name='stocker2003_extended_seaice_seesaw',
    tau_R=300.0, tau_S=1200.0, tau_A=100.0, tau_ANT=20.0,
    kappa=1.0, lambda_S=0.2, alpha=0.3, beta=0.2,
    gamma=4.0, delta=1.0, eta=0.2,
    T_S0=0.0, T_c=0.0, T_N=0.0,
    epsilon_R=0.0, epsilon_S=0.0, epsilon_A=0.0, epsilon_ANT=0.0,
    state_variables=['T_R','T_S','A','T_ANT'],
    diagnostic_variables=['T_N'])
```

**Parameters in `param_values`:** All 18 parameters. All taus validated > 0.

**Forcing:** Provides T_N(t). Falls back to `Tn` parameter if not provided.

**Additional methods:**
- `resolve_north(t, state)` — centralizes T_N lookup (forcing vs. parameter).
- `build_state_from_history(time, history)` — overrides parent to clip A to [0, 1] in the stored structured array post-solve.

**Hooks used:**
- `uses_post_history()` returns `True`
- `populate_diagnostics_from_history` reconstructs T_N
- `build_state_from_history` overridden for constraint enforcement

**Parameter contract:** `'legacy'` (default).

**Conformance:** Fully conformant. Demonstrates the correct pattern for post-solve constraint handling.

**Demo — `stocker2003_extended_seaice_demo.ipynb`:**
- Baseline (constant zero north forcing)
- DO-like north pulse (1500–3500 yr window) — transient T_R → T_S → A cascade
- Bistability: two initial A conditions (0.05 vs. 0.95) → different equilibria
- Timescale sensitivity: individual τ overrides, Antarctic response variation

---

### Lorenz 96

**What it models:** Lorenz (1996) N-variable chaotic system on a periodic ring; standard testbed for data assimilation and spatiotemporal chaos.

**Equation (component i):**
```
dx_i/dt = (x_{i+1} − x_{i-2}) · x_{i-1} − x_i + F(t)
```
Cyclic indexing mod N.

**Constructor:**
```python
Lorenz96(forcing=None, var_name='lorenz96', n=40, F=8.0,
         state_variables=['x0',...,'x{n-1}'], diagnostic_variables=[])
```

**Parameters in `param_values`:** `F`. Supports callables and `Forcing` objects.

**Forcing:** Provides a time-varying F(t). If provided, takes precedence over `F` parameter.

**Additional methods:**
- `_forcing_value(t, x)` — resolves forcing or F parameter. Necessary; no redundancy.

**Hooks used:** None.

**Parameter contract:** `'legacy'` (default).

**Conformance issues:**
- Appends state and time inside `dydt` (O(n²) repeated concatenation for long runs).

**Demo — `lorenz96_demo.ipynb`:**
- n=10 with constant F=8, RK45, `reframe_time_axis`
- Sinusoidal F(t) via `Forcing(lambda t: ...)` — time-varying forcing
- Space-time imshow visualization revealing wave-like spatiotemporal structure

---

### Lorenz 96 Two-Scale

**What it models:** Two-timescale extension of Lorenz-96: K global-scale X variables and J fast local-scale Y variables per X. Used for multiscale dynamics and data assimilation research.

**Equations:**
```
dX_k/dt = −X_{k-1}(X_{k-2} − X_{k+1}) − X_k + F − (hc/b)·Σ_j Y_{kj}
dY_{kj}/dt = −cb·Y_{k,j+1}(Y_{k,j+2} − Y_{k,j-1}) − c·Y_{kj} + (hc/b)·X_k
```

**Constructor:**
```python
Lorenz96TwoScale(forcing=None, var_name='lorenz96_two_scale',
    K=36, J=10, F=10.0, h=1.0, b=10.0, c=10.0,
    exact_rhs=False,  # True uses global roll, False uses per-block loop
    state_variables=['x0',...,'x{K-1}','y0',...,'y{KJ-1}'],
    diagnostic_variables=[])
```

**Parameters in `param_values`:** `F`, `h`, `b`, `c`.

**Additional methods:**
- `_forcing_value(t, x)` — resolves F.
- `_split_state(x)` — separates X (K) and Y (KJ) from state vector.
- `_state_to_arrays()` — reconstructs X (K × time) and Y (KJ × time) matrices.
- `integrate(t_span, y0, method, kwargs, run_name)` — **overrides parent**: adds `'l96_rk4'` method with fixed-step RK4 and internal sampling interval `si`.
- `run(si, total_time, y0, dt, method)` — convenience wrapper returning `(X, Y, t)` arrays directly.

**Hooks used:** None.

**Parameter contract:** `'legacy'` (default).

**Conformance issues:**
- Appends state and time inside `dydt`.
- Custom `integrate()` override that bypasses parent's solver selection logic, increasing maintenance burden.
- `'l96_rk4'` solver is hardcoded inside `integrate()`; no post-integrate hooks are called when using it.
- No dedicated demo notebook; only covered by the description notebook `L96-two-scale-description.ipynb`.

---

### Daisyworld

**What it models:** 0D Gaia-hypothesis model — black/white daisy fractions (Ab, Aw) and planetary temperature (T) regulate each other through albedo feedback under a varying stellar luminosity.

**Equations:**
```
A_planet = Aw·α_w + Ab·α_b + A_bare·α_g
T_w = T + q(A_planet − α_w)        [local white daisy temperature]
T_b = T + q(A_planet − α_b)        [local black daisy temperature]
β_w = max(0, 1 − β_width(T_opt − T_w)²)
β_b = max(0, 1 − β_width(T_opt − T_b)²)

dAw/dt  = Aw·(A_bare·β_w − γ)
dAb/dt  = Ab·(A_bare·β_b − γ)
dT/dt   = (S0·L_eff·(1 − A_planet)/4 − σT⁴) / C
```

Area fractions are normalized if Aw + Ab > 1.

**Constructor:**
```python
Daisyworld(forcing=None, var_name='daisyworld',
    alpha_w=0.75, alpha_b=0.25, alpha_g=0.5, gamma=0.3,
    q=20.0, T_opt=295.0, beta_width=0.003265,
    S0=1365.0, L=1.0, C=10.0, sigma=5.67051196e-8,
    state_variables=['Aw','Ab','T'],
    diagnostic_variables=['A_planet','A_bare','beta_w','beta_b'])
```

**Parameters in `param_values`:** All 11 parameters.

**Forcing:** Additive luminosity perturbation: `L_eff = L + forcing.get_forcing(t)`.

**Additional methods:**
- `_luminosity(t, x)` — combines base L with forcing perturbation.
- `_growth(T_local, t, x)` — computes growth rate (parabolic bell curve around T_opt).

**Hooks used:** None (`uses_post_history` not overridden, but diagnostics are appended in `dydt`).

**Parameter contract:** `'legacy'` (default).

**Conformance issues:**
- Appends state, all four diagnostics, and time inside `dydt`. The diagnostic appending is the most egregious case — it fires at every function evaluation (not just solution output points), so the diagnostic arrays are longer than the time axis for adaptive solvers.

**Demo — `daisyworld_demo.ipynb`:**
- Unforced (L=1) — equilibrium Aw, Ab, T
- Sinusoidal luminosity modulation (period 20, amplitude 0.08) — daisy tracking and thermal regulation
- Four-panel comparison: Aw, Ab, T, A_planet (unforced vs. forced)

---

## Insolation Utilities (`insolation.py`)

Not a model class. Provides two pure functions for computing solar radiation at the top of the atmosphere from orbital parameters:

| Function | Description |
|----------|-------------|
| `daily_insolation(lat, day, orb, S0, day_type, days_per_year)` | Daily-average insolation at a latitude |
| `instant_insolation(lat, day, lon, orb, S0, days_per_year)` | Instantaneous insolation at a lat/lon |

**Key parameters:** `lat` (degrees), `day` (calendar day or solar longitude per `day_type`), `orb` (dict with `'ecc'`, `'long_peri'`, `'obliquity'`), `S0` (solar constant, default 1365.2 W/m²).

**Implementation:** Berger (1978) eqn. 10. Fully vectorized; returns numpy array or xarray.DataArray depending on inputs. Depends on `climlab` for orbital parameter handling.

**Usage:** Generate forcing time series to pass into `pb.Forcing(...)` for orbital-driven models (e.g., EBM, G24). No dedicated demo notebook.

---

## Cross-Cutting Notes

### Models that have no dedicated demo notebook
- `Lorenz96TwoScale` — `L96-two-scale-description.ipynb` is the only notebook; it is a conceptual walkthrough, not a full integration demo
- `insolation.py` utilities — no notebook

### Models where diagnostics are longer than the time axis
EBM, Stommel, Daisyworld, and G24 all append diagnostics inside `dydt`. For adaptive solvers (RK45), the solver evaluates `dydt` at many intermediate points that are not in the output grid, so `diagnostic_variables` lists will be longer than `state_variables` and `time`. This is a latent bug for any code that assumes `len(diagnostics) == len(time)`.

### Models using `to_pyleo`
Demonstrated in: EBM demo (diagnostic stacking), Stocker 2003 Bipolar Seesaw demo (Ts series). Inherited by all models but rarely exercised.

### `time_util` lambda pattern (G24)
`Model3` uses a `time_util(t)` lambda to remap integration time to the time axis of the insolation data. This is a user-defined mapping set at instantiation; no built-in PBModel support. Other models may need a similar pattern when forcing data time and integration time differ in scale/offset.
