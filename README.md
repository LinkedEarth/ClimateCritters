# ClimateCritters
A menagerie of minimal paleoclimate models and the scars that taphonomy carves on their hides.

## Rationale
Much of the [strong inference](https://www.science.org/doi/10.1126/science.146.3642.347) in modern biology derives from the study of model organisms: non-human species extensively studied to understand fundamental biological phenomena. The findings often translate to other organisms, including humans. Climate science is bereft of organisms but rich in models. However, the emphasis is often on building the latest, greatest, most comprehensive model out there, which makes it challenging to understand behavior. Thus, despite [influential calls to explore this framework](https://doi.org/10.1175/BAMS-86-11-1609) (see also [Polvani et al, (2017)](https://eos.org/opinions/when-less-is-more-opening-the-door-to-simpler-climate-models)), model organisms (model models?) are still lacking in climate science. Another issue facing the study of past climates is that, prior to the instrumental era (CE 1850 or so), the records we have of them are often blurred, noisy, sparse and fragmentary. 

Because climate is capable of abrupt jumps, climate science pioneer Wally Broecker nicknamed it "The Angry Beast", and argued that our use of fossil fuels was akin to poking at this beast with sticks.  The purpose of `ClimateCritters` is to gather a collection of model "organisms" illuminating key aspects of climate dynamics (chaos, multiple equilibria, intermittency, tipping points), and how this behavior gets recorded in paleoclimate archives like ice or sediment cores. The core design principle is to code existing, simple models within a unified, object-oriented Python interface that makes it easy to experiment with those models, including:
- exploring model sensitivity via parameter sweeps or forcing scenarios
- exploring taphonomic effects like observational noise, bioturbation or age errors
- comparing the appropriateness of various timeseries analysis methods (e.g. causal analysis, tipping point detection) on well-understood models exhibiting nonlinear behavior.

## Climate Models
`ClimateCritters` presently gathers X models:

1. **Lorenz63**: Ed Lorenz's seminal paper, soberly titled [_Deterministic Nonperiodic Flow_](https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2) singlehandedly birthed chaos theory into existence. Though first intended to model Rayleigh-Bénard convection, this 3 equation, 3 variable model has become the paragon of a chaotic system. Now solvable nearly instantly using standard [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation) solvers, the Lorenz63 model is an incredibly versatile and acccessible tool to understand not only [weather prediction](https://doi.org/10.1175/1520-0477(1993)074<0049:ERAPAT>2.0.CO;2), but also the period-doubling route to chaos, or even [climate change](https://doi.org/10.1002/j.1477-8696.1993.tb05802.x).
2. **Ganopolski2024** [Ganopolski (2024)](https://doi.org/10.5194/cp-20-151-2024) proposed a minimal model to simulate glacial cycles ... (to be completed by Jordan)
3. **DO25**  [Melcher et al. (2025)](https://doi.org/10.5194/cp-21-115-2025)  proposed a conceptual model for Dansgaard–Oeschger event dynamics ..(to be completed by Maryam) 



## General Structure:

- **Signal**: this class will generate a pure (noise-free) series based on a given model. For instance model='Ganopolski2024.Model3' or  'Ganopolski2024.MiM'. Need to think about how to pass inititial conditions, boundary conditions, and model parameters. Giant dictionary exported to a yml file for traceability?  Signal should be a Pyleoclim series (trivial to export to csv, if one needs to). 
- **Noise**: this class will add various noise colors to the signals. Should also use pyleo.utils.tsmodel.random_time_index() to mimic the time sampling process, and create irregularly sampled series with reasonable characteristics.


## Time-varying parameters
Model parameters can be constants, callables, or `cc.core.Forcing` objects. This enables time-varying
or state-dependent parameters with a consistent API across models.

Example:
```python
lorenz = cc.signal_models.Lorenz63(
    forcing=cc.core.Forcing(lambda t: 0.0),
    sigma=lambda t, x, m: 10 + 2*np.sin(t/5),
    rho=cc.core.Forcing(lambda t: 28 + 5*np.sin(t/20)),
    beta=8/3,
)
```
