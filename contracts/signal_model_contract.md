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
param_values = {'flux': lambda t, state, model: model.get_param('k', t, state) * state[0]}
```

### 1.4 Accessing parameters inside `dydt`

Always use `self.get_param(name, t, state)` inside `dydt`. This resolves the value
regardless of whether it is a constant, callable, or `Forcing` object.

```python
def dydt(self, t, y):
    tau = self.get_param('tau', t, y)
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

*Sections to be added: state variable declaration, diagnostic variables, `dydt` structure,
`uses_post_history` path, naming conventions.*
