from __future__ import annotations

"""Generic box-model utilities for ClimateCritters.

This module provides a small declarative layer on top of :class:`CCModel` for
building simple ODE box models without writing a full bespoke subclass each
time.

Two styles are supported:

1. Explicit callable tendencies via :class:`BoxModelSpec.register_tendency`
2. Automatic box-network assembly via reciprocal exchange and directed
   transport relations

Examples
--------

Build a two-box carbon model with explicit tendencies:

>>> spec = BoxModelSpec("two_box")
>>> spec.register_state_variables(["A", "S"])
>>> spec.register_parameters(k=0.2, R=0.0, l_s=0.05, V_atm=1.0, V_surf=1.0)
>>> spec.register_input("R", fallback_param="R")
>>> def exchange(ctx):
...     return ctx.param("k") * (ctx["A"] / ctx.param("V_atm") - ctx["S"] / ctx.param("V_surf"))
>>> spec.register_relations({
...     "A": lambda ctx: -exchange(ctx) + ctx.input("R") - ctx.param("l_s") * ctx["A"],
...     "S": lambda ctx: exchange(ctx),
... })
>>> model = spec.make_boxmodel()

Build a mixed network with reciprocal exchange and directed transport:

>>> spec = BoxModelSpec("five_box")
>>> spec.register_state_variables(["SP", "DP", "SA", "DA", "SO"])
>>> spec.register_box_volumes(SP=8.0, DP=80.0, SA=5.0, DA=45.0, SO=20.0)
>>> spec.register_exchange("SP", "SA", 0.30)
>>> spec.register_exchange("SO", "SP", 0.16)
>>> spec.register_transport("SA", "DA", 0.22)
>>> spec.register_transport("DA", "SO", 0.12)
>>> model = spec.make_boxmodel()

"""

from dataclasses import dataclass

import numpy as np

from climatecritters.core.ccmodel import CCModel


@dataclass(frozen=True)
class _InputSpec:
    name: str
    fallback_param: str | None = None


@dataclass(frozen=True)
class _ExchangeSpec:
    left: str
    right: str
    rate_param: str


@dataclass(frozen=True)
class _TransportSpec:
    source: str
    target: str
    rate_param: str


class BoxModelContext:
    """Evaluation context for generic box-model tendencies and diagnostics.

    Parameters
    ----------
    model : GenericBoxModel
        The model instance currently being evaluated.
    t : float
        Current model time.
    state : array-like
        Current state vector ordered according to
        ``model.state_variables_names``.

    Notes
    -----
    The context exposes a compact, box-model-friendly API:

    - ``ctx["A"]`` for direct state lookup
    - ``ctx.param("k")`` for parameters resolved through ``CCModel.get_param``
    - ``ctx.input("R")`` for external prescribed inputs
    - ``ctx.volume("SP")`` and ``ctx.concentration("SP")`` for box-network
      models
    """

    def __init__(self, model, t, state):
        self.model = model
        self.t = float(t)
        self.state = np.asarray(state, dtype=float).reshape(-1)
        self._state_map = {
            name: float(self.state[i])
            for i, name in enumerate(self.model.state_variables_names)
        }

    def __getitem__(self, name):
        return self._state_map[name]

    def get(self, name, default=None):
        return self._state_map.get(name, default)

    def param(self, name):
        return self.model.get_param_value(name, self.t, self.state)

    def input(self, name):
        return self.model.resolve_input(name, self.t, self.state)

    def volume(self, name):
        return self.model.resolve_box_volume(name, self.t, self.state)

    def concentration(self, name):
        return self[name] / self.volume(name)


class BoxModelSpec:
    """Declarative specification for simple ODE box models.

    A ``BoxModelSpec`` stores state names, parameter defaults, optional
    inputs, diagnostics, and either:

    - explicit callable tendencies registered via
      :meth:`register_tendency` / :meth:`register_relations`, or
    - an automatic box network assembled from volumes, reciprocal exchange,
      and directed transport terms

    Parameters
    ----------
    name : str
        Model name used as the default ``var_name`` of the produced
        ``GenericBoxModel``.

    Notes
    -----
    Call :meth:`validate` (or :meth:`make_boxmodel`) before integrating.
    Validation checks that every state variable has a tendency relation
    (explicit mode) or a registered volume (automatic mode).

    See also
    --------
    GenericBoxModel : The ``CCModel`` subclass produced by :meth:`make_boxmodel`.
    BoxModelContext : Evaluation context passed to callable tendencies.

    Examples
    --------

    Minimal explicit one-box relaxation model:

    ```python
    import climatecritters as cc
    from climatecritters.signal_models.box_model import BoxModelSpec

    spec = BoxModelSpec("relaxation")
    spec.register_state_variables(["x"])
    spec.register_parameters(tau=10.0, x_eq=1.0)
    spec.register_tendency(
        "x", lambda ctx: (ctx.param("x_eq") - ctx["x"]) / ctx.param("tau")
    )
    model = spec.make_boxmodel()
    output = model.integrate(t_span=(0, 100), y0=[0.0], method='RK45')
    ```

    Automatic two-box exchange model:

    ```python
    spec = BoxModelSpec("exchange")
    spec.register_state_variables(["A", "S"])
    spec.register_box_volumes(A=1.0, S=50.0)
    spec.register_exchange("A", "S", 0.2)
    model = spec.make_boxmodel()
    ```

    """

    def __init__(self, name):
        self.name = name
        self.state_variables = []
        self.diagnostic_variables = []
        self.parameter_defaults = {}
        self.input_specs = {}
        self.tendency_relations = {}
        self.diagnostic_relations = {}
        self.volume_params = {}
        self.source_params = {}
        self.loss_params = {}
        self.exchange_relations = []
        self.transport_relations = []

    def register_state_variables(self, names):
        """Register prognostic state variables in integration order."""
        for name in names:
            if name not in self.state_variables:
                self.state_variables.append(str(name))
        return self

    def register_statevariables(self, names):
        return self.register_state_variables(names)

    def register_diagnostic_variables(self, names):
        """Register diagnostic variable names."""
        for name in names:
            if name not in self.diagnostic_variables:
                self.diagnostic_variables.append(str(name))
        return self

    def register_diagnosticvariables(self, names):
        return self.register_diagnostic_variables(names)

    def register_parameters(self, **parameters):
        """Register default parameter values.

        Values may be constants, callables, or ``Forcing``-like objects
        compatible with ``CCModel.get_param``.
        """
        self.parameter_defaults.update(parameters)
        return self

    def register_box_volumes(self, **volumes):
        """Register explicit box volumes for automatic network models.

        Parameters
        ----------
        volumes : dict
            Mapping from box name to box volume. These are stored internally as
            parameters named ``V__<box>``.
        """

        for box, value in volumes.items():
            box_name = str(box)
            param_name = f"V__{box_name}"
            self.volume_params[box_name] = param_name
            self.parameter_defaults[param_name] = value
        return self

    def register_source(self, box, value=0.0):
        """Register an external source term for a box in automatic networks.

        The source is added directly to the box tendency. ``value`` may be a
        constant or a time-varying parameter-compatible object.
        """
        box_name = str(box)
        param_name = f"source__{box_name}"
        self.source_params[box_name] = param_name
        self.parameter_defaults[param_name] = value
        return self

    def register_loss(self, box, value=0.0):
        """Register a first-order loss coefficient for a box.

        In automatic networks the tendency contribution is
        ``-loss__box * box_inventory``.
        """
        box_name = str(box)
        param_name = f"loss__{box_name}"
        self.loss_params[box_name] = param_name
        self.parameter_defaults[param_name] = value
        return self

    def register_input(self, name, fallback_param=None):
        """Register a prescribed input channel.

        Parameters
        ----------
        name : str
            Input name used by ``ctx.input(name)``.
        fallback_param : str or None
            Parameter name to use when the model has no ``forcing`` object.
        """
        self.input_specs[str(name)] = _InputSpec(str(name), fallback_param=fallback_param)
        return self

    def register_tendency(self, state_variable, relation):
        """Register a callable tendency for one state variable.

        ``relation`` must accept a :class:`BoxModelContext` and return the
        tendency for ``state_variable``.
        """
        self.tendency_relations[str(state_variable)] = relation
        return self

    def register_tendencies(self, relations):
        for name, relation in relations.items():
            self.register_tendency(name, relation)
        return self

    def register_relations(self, relations):
        """Register multiple callable tendencies at once."""
        return self.register_tendencies(relations)

    def register_diagnostic(self, name, relation):
        """Register a callable diagnostic evaluated from solved history."""
        self.diagnostic_relations[str(name)] = relation
        if name not in self.diagnostic_variables:
            self.diagnostic_variables.append(str(name))
        return self

    def register_exchange(self, left, right, rate, rate_param=None):
        """Register a reciprocal concentration-gradient exchange pathway.

        The flux is computed as

        ``rate * (C_left - C_right)``

        and applied with equal magnitude and opposite sign to the two boxes.
        """
        left_name = str(left)
        right_name = str(right)
        if rate_param is None:
            rate_param = f"kex__{left_name}__{right_name}"
        self.parameter_defaults[str(rate_param)] = rate
        self.exchange_relations.append(_ExchangeSpec(left_name, right_name, str(rate_param)))
        return self

    def register_transport(self, source, target, rate, rate_param=None):
        """Register a directed transport pathway.

        The transported mass flux is computed as

        ``rate * concentration(source)``

        and removed from ``source`` while being added to ``target``.
        """
        source_name = str(source)
        target_name = str(target)
        if rate_param is None:
            rate_param = f"q__{source_name}__{target_name}"
        self.parameter_defaults[str(rate_param)] = rate
        self.transport_relations.append(_TransportSpec(source_name, target_name, str(rate_param)))
        return self

    def uses_automatic_box_network(self):
        return bool(
            self.volume_params or
            self.source_params or
            self.loss_params or
            self.exchange_relations or
            self.transport_relations
        )

    def validate(self):
        """Validate that the spec is internally consistent."""
        if not self.state_variables:
            raise ValueError("BoxModelSpec requires at least one state variable.")

        extra = [name for name in self.tendency_relations if name not in self.state_variables]
        if extra:
            raise ValueError(f"Tendencies registered for unknown state variables: {extra}")

        if self.uses_automatic_box_network():
            missing_volumes = [name for name in self.state_variables if name not in self.volume_params]
            if missing_volumes:
                raise ValueError(
                    "Automatic box-network models require registered volumes for every "
                    f"state variable; missing {missing_volumes}"
                )

            known = set(self.state_variables)
            for rel in self.exchange_relations:
                if rel.left not in known or rel.right not in known:
                    raise ValueError(f"Exchange relation uses unknown boxes: {rel}")
            for rel in self.transport_relations:
                if rel.source not in known or rel.target not in known:
                    raise ValueError(f"Transport relation uses unknown boxes: {rel}")
            return

        missing = [name for name in self.state_variables if name not in self.tendency_relations]
        if missing:
            raise ValueError(f"Missing tendency relations for state variables: {missing}")

    def make_model(self, var_name=None, **parameter_overrides):
        """Instantiate a :class:`GenericBoxModel` from this spec."""
        self.validate()
        return GenericBoxModel(
            self,
            var_name=var_name if var_name is not None else self.name,
            **parameter_overrides,
        )

    def make_boxmodel(self, var_name=None, **parameter_overrides):
        return self.make_model(var_name=var_name, **parameter_overrides)


class GenericBoxModel(CCModel):
    """``CCModel`` subclass produced by :class:`BoxModelSpec`.

    Users construct this via :meth:`BoxModelSpec.make_boxmodel` rather than
    instantiating it directly.  The resulting model integrates with
    ``model.integrate(...)``, exports to Pyleoclim via ``output.to_pyleo()``,
    and computes diagnostics from solved history via
    ``populate_diagnostics_from_history``.

    Parameters
    ----------
    spec : BoxModelSpec
        The validated specification object.
    var_name : str or None
        Override for the model name.  Defaults to ``spec.name``.
    kwargs : dict
        Additional parameter overrides applied on top of
        ``spec.parameter_defaults``.
    """

    def __init__(self, spec, var_name=None, *args, **kwargs):
        self.spec = spec
        kwargs.pop("parameter_contract", None)
        param_values = dict(spec.parameter_defaults)
        param_values.update(kwargs)

        super().__init__(
            variable_name=var_name if var_name is not None else spec.name,
            state_variables=list(spec.state_variables),
            diagnostic_variables=list(spec.diagnostic_variables),
            *args,
        )

        self.param_values = dict(param_values)
        self.params = ()
        for name, value in self.param_values.items():
            setattr(self, name, value)

    def uses_post_history(self):
        """Route diagnostics through CCModel's post-history hooks."""
        return True

    def resolve_input(self, name, t, state):
        """Resolve a registered input from its fallback parameter.

        To drive an input with a time-varying external signal, register a
        forcing on the corresponding fallback parameter::

            model.register_forcing('fallback_param_name', forcing_obj)
        """
        if name not in self.spec.input_specs:
            raise KeyError(f"Input '{name}' is not registered on this BoxModelSpec.")

        input_spec = self.spec.input_specs[name]
        if input_spec.fallback_param is None:
            raise ValueError(f"Input '{name}' has no fallback parameter defined.")
        return self.get_param_value(input_spec.fallback_param, t, state)

    def resolve_box_volume(self, name, t, state):
        """Resolve the explicit volume of a registered box."""
        if name not in self.spec.volume_params:
            raise KeyError(f"No registered volume for box '{name}'.")
        value = float(self.get_param_value(self.spec.volume_params[name], t, state))
        if value <= 0.0:
            raise ValueError(f"Volume for box '{name}' must be > 0.")
        return value

    def _context(self, t, state):
        return BoxModelContext(self, t, state)

    def _automatic_tendency(self, box_name, ctx):
        """Assemble one box tendency from sources, losses, exchange, and transport."""
        total = 0.0
        if box_name in self.spec.source_params:
            total += float(self.get_param_value(self.spec.source_params[box_name], ctx.t, ctx.state))
        if box_name in self.spec.loss_params:
            total -= float(self.get_param_value(self.spec.loss_params[box_name], ctx.t, ctx.state)) * ctx[box_name]

        for rel in self.spec.exchange_relations:
            rate = float(self.get_param_value(rel.rate_param, ctx.t, ctx.state))
            flux = rate * (ctx.concentration(rel.left) - ctx.concentration(rel.right))
            if box_name == rel.left:
                total -= flux
            elif box_name == rel.right:
                total += flux

        for rel in self.spec.transport_relations:
            rate = float(self.get_param_value(rel.rate_param, ctx.t, ctx.state))
            flux = rate * ctx.concentration(rel.source)
            if box_name == rel.source:
                total -= flux
            elif box_name == rel.target:
                total += flux

        return total

    def dydt(self, t, state):
        """Evaluate the model tendency vector at time ``t`` and state ``state``."""
        ctx = self._context(t, state)
        if self.spec.uses_automatic_box_network():
            return [
                float(self._automatic_tendency(name, ctx))
                for name in self.state_variables_names
            ]
        return [float(self.spec.tendency_relations[name](ctx)) for name in self.state_variables_names]

    def populate_diagnostics_from_history(self, time, history):
        """Populate diagnostics by replaying callable definitions over solved history."""
        diagnostics = {}
        for name in self.spec.diagnostic_variables:
            if name in self.spec.diagnostic_relations:
                diagnostics[name] = np.asarray(
                    [
                        float(self.spec.diagnostic_relations[name](self._context(t, row)))
                        for t, row in zip(time, history)
                    ],
                    dtype=float,
                )
            elif name in self.spec.input_specs:
                diagnostics[name] = np.asarray(
                    [float(self.resolve_input(name, t, row)) for t, row in zip(time, history)],
                    dtype=float,
                )
            else:
                diagnostics[name] = np.full(len(time), np.nan, dtype=float)
        self.diagnostic_variables = diagnostics
