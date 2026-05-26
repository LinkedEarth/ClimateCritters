# PaleoBeasts Docstring Style Guide

Reference style: [pyleoclim `GeoSeries`](https://github.com/LinkedEarth/Pyleoclim_util/blob/master/pyleoclim/core/geoseries.py).
Format: **NumPy / numpydoc** (`numpy` Sphinx extension).

---

## 1. General rules

- Use triple double-quotes `"""` for every docstring.
- The **first line is a one-sentence summary** ending in a period, on the same line as the opening `"""`.
- Leave a **blank line** after the summary if the docstring has more content.
- Section underlines use hyphens repeated to match the section name length.
- Section order (omit sections that don't apply):

  1. Summary sentence
  2. Extended description (optional prose)
  3. Parameters
  4. Returns
  5. Raises
  6. Notes
  7. See also
  8. Examples

---

## 2. Classes

The class docstring lives on the class body (not `__init__`).

### 2a. Template

```python
class MyModel(PBModel):
    """One-sentence description of the model.

    Optional one-to-three sentence elaboration — governing equation,
    key assumptions, what makes this variant distinct.

    May include a rendered math block:

        C dT/dt = F_in - F_out

    Parameters
    ----------
    forcing : pb.core.Forcing
        Description of what this forcing represents and its units.
    param_a : float or callable or pb.core.Forcing
        Description.  Default is X.
    param_b : int, optional
        Description.  Default is Y.

    Notes
    -----
    Any implementation contracts, edge cases, or behavioral invariants
    a user needs to know.  Cross-reference the callable-parameter
    contract when relevant::

        Callables must follow contracts/signal_model_contract.md.

    See also
    --------
    pb.core.PBModel : Base class.
    OtherModel : Related model.

    Examples
    --------
    .. code-block:: python

        import paleobeasts as pb
        from paleobeasts.signal_models.ebm import MyModel

        forcing = pb.core.Forcing(lambda t: 1360.0)
        model = MyModel(forcing=forcing, param_a=4.0)
        output = model.integrate(t_span=(0, 1000), y0=[288.0], method='RK45')

    """
```

### 2b. Key points

| Rule | Rationale |
|------|-----------|
| Include the governing equation as an indented code block | Readers understand the physics at a glance |
| List every `__init__` parameter (except `self`) | pyleoclim puts all parameter docs on the class, not `__init__` |
| For `float or callable or pb.core.Forcing` params, always note the default | Avoids source-diving |
| Put contract/invariant details in **Notes**, not the summary | Keeps the summary scannable |

---

## 3. Methods and functions

### 3a. Simple one-liner (delegators, thin wrappers)

```python
def calc_OLR(self, T, t):
    """Return OLR at state T and time t (delegates to param_values['OLR'])."""
    ...
```

Use a one-liner when the function does exactly one obvious thing with no non-trivial parameters.

### 3b. Full docstring template

```python
def my_method(self, T, t):
    """Short imperative summary ending in a period.

    Optional extended description (1–3 sentences) if the summary
    alone is insufficient.

    Parameters
    ----------
    T : float or array-like
        Current temperature state.
    t : float
        Current time.  Unused if the method is time-independent;
        kept for a uniform external signature.

    Returns
    -------
    result : float or ndarray
        Description of what is returned and its units.

    Notes
    -----
    Anything subtle: overridden behaviour, side-effects, units
    assumptions, or references to the callable-parameter contract.
    """
    ...
```

### 3c. `dydt` methods

`dydt` has a standard structure shared across all PBModel subclasses.
Always document its **side effects** (appending to internal accumulators)
and whether the model sets `uses_post_history = True`.

```python
def dydt(self, t, x):
    """Evaluate the right-hand side of the ODE at time t and state x.

    Called by the solver at each timestep.  Appends to
    ``self.state_variables`` and ``self.diagnostic_variables`` as a
    side effect (step-by-step models only; models with
    ``uses_post_history = True`` do not accumulate state here).

    Parameters
    ----------
    t : float
        Current time.
    x : array-like
        Current state vector.  Length must match the number of
        integrated state variables.

    Returns
    -------
    dydt : list of float
        Time-derivatives in the same order as the state vector.
    """
    ...
```

---

## 4. Section reference

### Parameters

```
Parameters
----------
name : type
    Description.  Default is X.
```

- `type` should be specific: `float`, `int`, `array-like`, `str`,
  `callable`, `pb.core.Forcing`, `bool`, or a union like
  `float or callable or pb.core.Forcing`.
- Append `Default is X.` to the description (not a separate line).
- For optional parameters, write `name : type, optional`.

### Returns

```
Returns
-------
name : type
    Description including units where relevant.
```

If there is only one return value, the `name` can be a meaningful noun
(e.g., `albedo`, `temperature`, `func`).

### Notes

Use for:
- Implementation contracts (callable signatures, `param_values` keys)
- Non-obvious physical assumptions (e.g. annual-mean insolation)
- Pointers to the math derivation or reference paper
- Behavioral flags like `uses_post_history`

### Examples

PaleoBeasts uses `.. code-block:: python` until Sphinx `jupyter-execute`
is configured.  Show the minimal working path: construct the model,
integrate, access output.

```rst
Examples
--------

.. code-block:: python

    import paleobeasts as pb
    from paleobeasts.signal_models.ebm import EBM0D

    forcing = pb.core.Forcing(lambda t: 1360.0)
    model = EBM0D(forcing=forcing)
    output = model.integrate(t_span=(0, 500), y0=[288.0], method='RK45')
    output.plot('T')
```

---

## 5. PaleoBeasts-specific conventions

### 5a. Callable-parameter contract

Whenever a parameter accepts a callable, mention the contract:

```
param : float or callable or pb.core.Forcing
    Heat capacity.  If callable, must follow the signature contract in
    ``contracts/signal_model_contract.md``: ``(t)``, ``(t, state)``,
    or ``(t, state, model)`` with the first argument named ``t`` or
    ``time``.  Default is 4.0.
```

### 5b. param_values

Note which parameters are registered in `param_values` and therefore
time-vary-able through `get_param_value`.  This is best placed in **Notes**
for the class docstring.

### 5c. Governing equation

Always include the governing differential equation in the class docstring
as an indented code block.  Use standard notation: `dT/dt`, `div(grad T)`,
etc.  Include units for every term in Notes if the equation is non-trivial.

### 5d. Grid-aware methods (1D models)

Spatial methods should note the coordinate system in use
(e.g. `x = sin(phi)`) and the boundary conditions applied.

### 5e. Overridden base-class methods

When a method overrides a base-class method, say so explicitly in the
summary or in Notes:

```
"""Array ice-albedo: step function with linear transition.

Overrides ``EBMBase.calc_albedo``.
"""
```

---

## 6. Quick reference: what a complete EBM class looks like

```
Class docstring
  ├── Summary sentence
  ├── Extended description (governing equation)
  ├── Parameters (all __init__ args except self)
  ├── Notes (uses_post_history flag, callable contract, param_values)
  ├── See also (PBModel, related models)
  └── Examples (construct → integrate → output)

calc_* helpers
  ├── One-liner if they just delegate
  └── Full docstring (Parameters, Returns, Notes) if they do real work

dydt
  └── Always full docstring with Parameters, Returns, side-effect note

populate_diagnostics_from_history (if present)
  └── Full docstring: what it populates and from where

Module-level functions (OLR_func, albedo_func, …)
  ├── Summary sentence
  ├── Parameters
  └── Returns (including that OLR_func returns a callable)
```
