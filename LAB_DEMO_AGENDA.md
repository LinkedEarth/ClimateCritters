# PaleoBeasts — Lab Group Demo Agendas

Two versions of a lab group introduction, depending on available time.

**Audience:** Graduate students and postdocs familiar with ODEs and paleoclimate concepts,
varying Python depth.  
**Goal:** Everyone leaves able to run and modify a model on their own.  
**Environment:** All notebooks require the `pb_jpl_env` conda environment.

---

## 45-Minute Version — Demo-First

**One guiding principle:** show the full pipeline — model → noise → proxy-like data — once,
end to end, and make it feel effortless. Skip architecture; lead with impact.

**Prep:** Pre-run and save outputs for `Ganopolski2024_demo` and `model_noise_demo` before
the session. Both have cells that take 10–20 seconds. Show saved outputs for slow cells;
run the fast interactive cells live.

---

### Block 1 — Hook (5 min)
**Format:** one notebook, no slides

Open `lorenz63.ipynb` and run the butterfly attractor cell cold. Say one sentence:
*"This is a three-line ODE, and the interface for running a G24 ice-volume model is
identical."* Then close it.

Goal: establish that the package handles serious nonlinear systems and is not complicated
to use. That is all this block needs to do.

---

### Block 2 — A model your group actually cares about (15 min)
**Notebook:** `notebooks/model_demos/Ganopolski2024_demo.ipynb`

Run live, narrate as you go:

1. Instantiate with synthetic forcing, integrate, plot — show that the 100 kyr cycles
   emerge. (~3 min)
2. Swap in real orbital insolation with `Forcing.from_csv(dataset='insolation')`.
   Emphasise this is **one line**. (~2 min)
3. Run the timestep sensitivity sweep. Let the figure speak — the point that `dt` matters
   for regime-switching models lands visually without explanation. (~5 min)
4. Show `set_param()` briefly: *"You can modify any parameter or function on a live
   instance without touching the class."* Don't demo it deeply; just show the line
   exists. (~2 min)

Leave ~3 minutes for questions — someone will ask about the orbital forcing data or
the parameter choices.

---

### Block 3 — The pipeline that connects to their work (15 min)
**Notebook:** `notebooks/functionality_demos/model_noise_demo.ipynb`

Run the whole notebook straight through, narrating:

- *"Here's the clean model output."*
- *"Here's what it looks like after we add AR(1) noise with realistic autocorrelation."*
- *"Here's what it looks like after exponential downsampling — this is your sediment core."*

The three-panel figure at the end is the money shot. Most people in a palaeoclimate lab
will immediately recognise their own workflow in it.

Then show the single line that hands everything to pyleoclim: `m.to_pyleo()`. If anyone
in the room already uses pyleoclim, this lands hard.

---

### Block 4 — What else is in here (5 min)
**Format:** `PACKAGE_SUMMARY.md` on screen — scroll, don't demo

Point out:

- The model table: *"There's a Stommel THC model, a 1D latitudinal EBM, a Daisyworld,
  a bipolar seesaw — all the same interface."*
- The noise utilities: *"You can also run these on real proxy series, not just model
  output."*
- The pendulum notebook: *"There's a pendulum demo that covers chaos, strange attractors,
  and how to subclass a model — good for onboarding to the framework."*

Goal: give a map, not a tour.

---

### Block 5 — Run something together (5 min)
**Format:** live code

Pick one model relevant to a specific person in the room — ask before the session — and
run it from scratch. The whole cycle takes about 60 seconds:

```python
from paleobeasts.signal_models import Stommel
m = Stommel()
m.integrate(t_span=(0, 40), y0=[1.0, 0.5])
```

Change one parameter and rerun. The point is not the result — it is showing that
iteration is fast enough to do interactively.

---

### What to cut vs. the 75-minute version

| 75-min topic | Cut entirely | Compressed to a mention |
|---|:---:|:---:|
| Architecture slides | ✓ | |
| Forcing deep-dive | ✓ | |
| Pendulum notebook | ✓ | |
| BoxModelSpec | ✓ | |
| Extensibility / subclassing | ✓ | |
| Energy / tolerance discussion | ✓ | |
| `model_noise_demo` | | ✓ (run it, don't explain it) |
| `lorenz63` | | ✓ (hook only, ~2 min) |
| G24 demo | | full (15 min) |

---
---

## 75–90 Minute Version — Demo and Tutorial

**Goal:** Understanding as well as exposure. Attendees should leave knowing *why* the
interface works the way it does, not just that it works.

**Prep:** Same as above — pre-run the slow cells in G24 and noise notebooks.
Have a pre-run fallback for every notebook in case of environment issues.

---

### Block 1 — Why this exists (10 min)
**Format:** slides

The key question to answer upfront: *"What does PaleoBeasts give me that scipy/numpy
doesn't?"*

- **The workflow problem** — the steps from conceptual model to publishable output
  (integrate → add noise → downsample → compare to proxy data) are the same for almost
  every model, but everyone re-implements them from scratch. Show a cartoon of this
  pipeline.
- **What the package standardises** — unified `PBModel` interface, `Forcing` abstraction,
  pyleoclim handoff. One diagram of `integrate → to_pyleo() → pyleoclim analysis` is
  worth a lot here.
- **What it does not do** — it is not a solver library, not a data repository, not a
  fitting framework. Scope-setting avoids confusion later.

---

### Block 2 — Core architecture tour (10 min)
**Format:** slides + one live code cell

- One slide on `PBModel`: three things every model has — `dydt`, `forcing`,
  `param_values`. Show the inheritance diagram (PBModel → EBM, G24, Stommel, etc.).
- One slide on `Forcing`: the three input forms (callable, array, element sequence).
  Key insight to land: *any parameter can also be a Forcing object*, so the same model
  handles constant, sinusoidal, and orbital-data forcing without code changes.
- **One live cell** (~2 min) — before any deep explanation, show the minimal
  integrate-and-plot pattern:

```python
from paleobeasts.signal_models import SimplePendulum
m = SimplePendulum(L=1.0, g=9.81, damping=0.5)
m.integrate(t_span=(0, 20), y0=[1.5, 0], method='RK45')
plt.plot(m.time, m.state_variables['theta'])
```

Point: *this is all you need to run any model.* The interface is the same whether you
are running a pendulum or an ice-volume model.

---

### Block 3 — Signal models tour (20 min)
**Format:** two notebooks

Pick two models that form a contrast.

**3a. `lorenz63.ipynb` — dynamical systems warm-up (~10 min)**

Most people in a palaeoclimate lab have heard of chaos but have not played with it
interactively. Focus on Sections 1 and 2 only (canonical run, butterfly attractor,
sensitive dependence on ICs). Skip the time-varying parameters section unless someone
asks. Key takeaway: *sensitive dependence on ICs is relevant to any nonlinear
palaeoclimate model.*

**3b. `Ganopolski2024_demo.ipynb` — a research-grade model (~10 min)**

Most relevant to the group's actual work. Focus on:
- The timestep sensitivity sweep — immediately actionable
- The `from_csv(dataset='insolation')` call — real orbital forcing in one line
- Skip the derivative-function details on first pass

The contrast between the two notebooks makes the point that the *same interface* covers
both toy systems and literature models.

---

### Block 4 — Forcing and time-varying parameters (10 min)
**Format:** live code

Run this live rather than showing a pre-executed notebook — the interactivity makes the
flexibility tangible. Show three versions of the same model, three forcing styles:

```python
from paleobeasts.core import Forcing
from paleobeasts.signal_models import EBM
import numpy as np

f_sine    = Forcing(lambda t: 1361 + 5 * np.sin(2 * np.pi * t / 11))
f_orbital = Forcing.from_csv(dataset='insolation')
f_const   = Forcing(1361.0)

for f, label in [(f_sine, 'sinusoidal'), (f_orbital, 'orbital'), (f_const, 'constant')]:
    m = EBM(forcing=f)
    m.integrate(t_span=(0, 500), y0=[285.0])
    plt.plot(m.time, m.state_variables['T'], label=label)
```

Then show a time-varying *parameter* (e.g. albedo as a function of temperature) to
drive home that parameters and forcing are treated symmetrically.

---

### Block 5 — The realistic data pipeline (10 min)
**Notebook:** `notebooks/functionality_demos/model_noise_demo.ipynb`

Run straight through and narrate:

1. Run G24 → get a clean ice volume signal.
2. `to_pyleo()` → hand it to pyleoclim.
3. Add AR(1) noise (show how τ and σ are chosen).
4. Downsample exponentially (this represents sediment core sampling).
5. Plot all three on one axis.

Key talking points:
- The noise and resampling modules work on real proxy data too, not just model output.
- `from_series()` in `noise.py` can match the autocorrelation of an observed series,
  so surrogates are statistically consistent with the data.

---

### Block 6 — Extensibility: where do I put my model? (10 min)
**Format:** one slide + live code

- **One slide:** the three ways to extend the package — subclass `PBModel` for a new
  model, use `BoxModelSpec` for a multi-box system, use `set_function()` for a one-off
  instance modification.
- **Live demo (~5 min):** show the `DrivenDoublePendulum` from
  `notebooks/model_demos/pendulum_demo.ipynb`. It is a short, clean subclass that adds
  a cosine drive term in about five lines by calling `super().dydt()` and mutating the
  result. This is the template for adding a new forcing term or feedback to any existing
  model.
- **Point to `BoxModelSpec`** for anyone who works with carbon cycling or multi-reservoir
  systems — the declarative spec approach lets them prototype a 5-box model in ~20 lines.

---

### Wrap-up — What to try first (5 min)
**Format:** slide or printed handout

| Research focus | First notebook to run | First thing to try |
|---|---|---|
| Ice ages / orbital forcing | `Ganopolski2024_demo` | Adjust `vc` or `f1`; watch how interglacials shift |
| Ocean circulation | `stommel_demo` or `two_box_carbon_demo` | Add sinusoidal forcing; watch overturning `q` |
| Proxy interpretation | `model_noise_demo` | Swap in your own AR(1) parameters |
| Dynamical systems | `lorenz63` or `pendulum_demo` | Add time-varying parameters; look for regime shifts |
| New model | `two_box_carbon_demo` (BoxModelSpec section) | Add a third box |

---

### Notes for both sessions

- Avoid `reference_content/taxonomy_entry.py` — it is specialised enough to derail into
  a separate conversation.
- The `two_box_carbon_demo.ipynb` has a mass-conservation verification cell
  (`total drift < 1e-6`) that is worth a quick mention for audiences who care about
  numerical rigour, even if you do not demo the full notebook.
- The `solver_choice.ipynb` notebook is a good pointer for anyone who asks *"why does my
  regime-switching model give different results at different timesteps?"* — send them
  there after the session rather than opening it during.
