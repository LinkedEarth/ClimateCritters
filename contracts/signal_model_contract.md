# Signal Model Contract

This document defines the conventions for constructing signal models in PaleoBeasts.
Each section is a binding rule, not a suggestion. New models and tests must follow these
conventions; deviations should be discussed and reflected here before being introduced.

---

## 1. Parameter Callables

Parameters in `param_values` may be constants, callables, or `Forcing` objects. This
section governs the callable case.

### 1.1 Supported signatures

A callable parameter must have one of these three signatures, where the first argument
is always time and must be named `t` or `time`:

| Signature | Use when the parameter depends on… |
|---|---|
| `(t)` | time only |
| `(t, state)` | time and the current state vector |
| `(t, state, model)` | time, state, and the model instance |

Any other form — reversed argument order, model-first, state-only, zero-argument — is not
supported.

### 1.2 What each argument receives

| Argument | Type | Description |
|---|---|---|
| `t` | `float` | Current integration time |
| `state` | `ndarray` | Current state vector (same shape as `y0`) |
| `model` | `PBModel` subclass | The model instance; use for accessing other parameters or attributes |

### 1.3 Examples

```python
# Constant — no callable needed
param_values = {'tau': 1500.0}

# Time-varying only
param_values = {'tau': lambda t: 1500 + 200 * np.sin(2 * np.pi * t / 41)}

# Time- and state-dependent
param_values = {'alpha': lambda t, state: 0.3 if state[-1] > 273 else 0.6}

# Needs model context (e.g., to access another resolved parameter)
param_values = {'flux': lambda t, state, model: model.get_param_value('k', t, state) * state[0]}
```

### 1.4 Accessing parameters inside `dydt`

Always use `self.get_param(name, t, state)` inside `dydt`. This resolves the value
regardless of whether it is a constant, callable, or `Forcing` object.

```python
def dydt(self, t, y):
    tau = self.get_param_value('tau', t, y)
    ...
```

### 1.5 Rationale

Earlier versions of PaleoBeasts supported heuristic dispatch that inferred argument
order from parameter names (e.g. `ebm_model`, `self`, reversed `(state, t)`). This
made the base class dependent on naming conventions in individual models and produced
subtle, silent bugs when names didn't match expectations. The strict convention above
eliminates all guesswork: arity and the name of the first argument are the only things
that matter.

---

## 2. State and Diagnostic Variable Declaration

### 2.1 State variables

Declare all state variables in `__init__` by passing a list of names to `super().__init__`:

```python
super().__init__(
    forcing,
    variable_name,
    state_variables=['T', 'S'],
    diagnostic_variables=['q'],
)
```

The order of `state_variables` defines the order of the state vector `y` passed to `dydt`
and must match the order of values in `y0`.

### 2.2 Non-integrated state variables

Some models track a discrete or regime variable that is not integrated by the solver (e.g.
a mode switch). Declare these in `non_integrated_state_vars`; they are excluded from the
vector passed to the solver but included in the structured output array.

This pattern couples `dydt` to the model's own accumulation buffers and should not be
used for new models. Prefer encoding all state in the integrated vector.

### 2.3 Diagnostic variables

Declare diagnostic variable names in `diagnostic_variables`. For `uses_post_history = True`
models, populate them by overriding `populate_diagnostics_from_history`. For step-by-step
models, append to `self.diagnostic_variables[name]` inside `dydt`.

### 2.4 `param_values`

Every model must define a `param_values` dict in `__init__` mapping parameter names to
their default values (constants, callables, or `Forcing` objects). Set `self.params = ()`.

```python
self.param_values = {'tau': tau, 'beta': beta}
self.params = ()
```

---

## 3. `dydt` and `uses_post_history`

### 3.1 The `dydt` contract

`dydt(self, t, y)` must return a list of derivatives with the same length as the
**integrated** portion of the state vector. It is called by the solver repeatedly and
must not rely on call order for correctness.

```python
def dydt(self, t, y):
    x = y[0]
    tau = self.get_param_value('tau', t, y)
    return [-x / tau]
```

### 3.2 `uses_post_history`

Set `uses_post_history = True` as a **class attribute** on any model whose `dydt` is
free of side effects. This is the preferred pattern for all new models.

```python
class MyModel(PBModel):
    uses_post_history = True

    def dydt(self, t, y):
        ...  # pure: no appending to self.state_variables or self.diagnostic_variables
```

When `uses_post_history = True`, the solver accumulates the full trajectory internally
and passes it to `post_integrate` after the solve. Override
`populate_diagnostics_from_history(time, history)` to derive any diagnostics from the
complete trajectory:

```python
def populate_diagnostics_from_history(self, time, history):
    self.diagnostic_variables['flux'] = np.diff(history[:, 0], prepend=history[0, 0])
```

### 3.3 Side effects in `dydt` (legacy pattern)

Models with `uses_post_history = False` (the default) may append to
`self.state_variables`, `self.time`, and `self.diagnostic_variables` inside `dydt`.
This pattern is retained for models with discrete state transitions that must be read
back on the next step. It should not be used for new models; use `uses_post_history = True`
instead.

---

## 4. Integration Output (`PBOutput`)

### 4.1 `integrate()` returns a `PBOutput`

```python
output = model.integrate(t_span=(0, 4000), y0=[0.0], method='euler', kwargs={'dt': 10.0})
```

`PBOutput` carries the full trajectory and provides output-focused operations. The model
also stores the latest output as `model.output` for backward-compatible attribute access,
but capturing the return value is preferred when running multiple experiments.

### 4.2 Time axes

| Attribute | Content | Mutability |
|---|---|---|
| `output.model_time` | Raw time axis from the solver | Never modified |
| `output.time` | Current user-facing time axis | Updated by `reframe_time_axis` |

At construction `time is model_time`. After `reframe_time_axis`, `time` reflects the
requested grid while `model_time` is unchanged.

### 4.3 `output_time` — automatic windowing

Pass `output_time` to `integrate()` to place the output on a specific grid immediately,
for example to exclude a spin-up period:

```python
output = model.integrate(
    t_span=(0, 5000),
    y0=[0.0],
    output_time=np.linspace(1000, 5000, 401),
)
# output.time  → 1000–5000
# output.model_time → full solver grid
```

The window can always be changed later — including extending back into the excluded
transient — by calling `output.reframe_time_axis(new_time)` again, because the full
solver trajectory is retained in `output.solution`.

### 4.4 Post-hoc noise

Add noise to output variables after integration using `output.add_noise`. The clean
values are saved automatically so `output.remove_noise` can restore them. Each
`PBOutput` instance tracks its own noise state independently, making it straightforward
to generate multiple noisy realizations from a single deterministic run:

```python
output = model.integrate(...)
noisy_outputs = []
for noise in ensemble_of_noise_series:
    import copy
    o = copy.deepcopy(output)
    o.add_noise('Ts', noise)
    noisy_outputs.append(o)
```

---

*Sections to be added: naming conventions.*
