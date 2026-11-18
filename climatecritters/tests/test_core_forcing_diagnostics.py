''' Tests that diagnostics computed after integration see registered parameter forcings

Diagnostics built in ``populate_diagnostics_from_history`` are evaluated
outside the solver, so they must apply parameter forcings themselves
(``Model.forcings_applied``). The check: a model built with ``p = v`` and a
model built with the default ``p`` plus a constant replacement forcing
``p -> v`` follow the same trajectory, so their diagnostics must agree.

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}
'''

import numpy as np
import pytest
import climatecritters as cc

from climatecritters.model_critters import BoxModelSpec
from climatecritters.model_critters.damped_spring import DampedSpring
from climatecritters.model_critters.ebm import EBM0D
from climatecritters.model_critters.melcher25 import Melcher25
from climatecritters.model_critters.pendulum import (
    DoublePendulum, DrivenPendulum, MultiPendulumBeta, PendulumRodBeta, SimplePendulum,
)
from climatecritters.model_critters.stocker2003_bipolar_seesaw import (
    Stocker2003BipolarSeesaw, Stocker2003ExtendedSeaIceSeesaw,
)
from climatecritters.model_critters.two_box_carbon import TwoBoxCarbon


def _multi_pendulum(**kwargs):
    rods = [PendulumRodBeta(m=1.0, L=1.0), PendulumRodBeta(m=0.5, L=0.8)]
    return MultiPendulumBeta(rods=rods, **kwargs)


def _stocker_box(Tn=0.0):
    spec = BoxModelSpec("stocker_box")
    spec.register_statevariables(["Ts"])
    spec.register_diagnosticvariables(["Tn"])
    spec.register_parameters(tau=1000.0, beta=-1.0, Tn=Tn)
    spec.register_input("Tn", fallback_param="Tn")
    spec.register_relations({
        "Ts": lambda ctx: (ctx.param("beta") * ctx.input("Tn") - ctx["Ts"]) / ctx.param("tau"),
    })
    return spec.make_boxmodel()


# (id, factory, parameter, forced value, y0, t_span)
CASES = [
    ('Stommel', cc.Stommel, 'k', 2.0, [1.0, 0.0], (0, 5)),
    ('DampedSpring', DampedSpring, 'k', 2.0, [1.0, 0.0], (0, 5)),
    ('SimplePendulum', SimplePendulum, 'g', 5.0, [0.5, 0.0], (0, 5)),
    ('DrivenPendulum', DrivenPendulum, 'A', 0.5, [0.5, 0.0], (0, 5)),
    ('DoublePendulum', DoublePendulum, 'g', 5.0, [0.5, 0.0, 0.3, 0.0], (0, 2)),
    ('MultiPendulumBeta', _multi_pendulum, 'g', 5.0, [0.3, 0.0, 0.2, 0.0], (0, 2)),
    ('Stocker2003BipolarSeesaw', Stocker2003BipolarSeesaw, 'Tn', 1.0, [0.0], (0, 500)),
    ('Stocker2003ExtendedSeaIceSeesaw', Stocker2003ExtendedSeaIceSeesaw, 'T_N', 1.0,
     [0.0, 0.0, 0.3, 0.0], (0, 500)),
    ('TwoBoxCarbon', TwoBoxCarbon, 'R', 1.0, [100.0, 50.0], (0, 20)),
    ('GenericBoxModel', _stocker_box, 'Tn', 1.0, [0.0], (0, 500)),
    ('Melcher25', Melcher25, 'alpha', 0.6, [1.0, 0.0], (0, 20)),
    ('Daisyworld', cc.Daisyworld, 'alpha_g', 0.45, [0.2, 0.2, 295.0], (0, 100)),
    ('EBM0D', EBM0D, 'S0', 1300.0, [288.0], (0, 50)),
]


class TestCoreForcingDiagnostics:
    @pytest.mark.parametrize('name, factory, param, value, y0, t_span', CASES,
                             ids=[c[0] for c in CASES])
    def test_diagnostics_see_param_forcing_t0(self, name, factory, param, value, y0, t_span):
        direct = factory(**{param: value})
        out_direct = direct.integrate(t_span=t_span, y0=y0, method='RK45')

        forced = factory()
        forced.register_forcing(param, cc.Forcing(lambda t: value + 0 * t))
        out_forced = forced.integrate(t_span=t_span, y0=y0, method='RK45')

        # same dynamics...
        for v in out_direct.state_variable_names:
            np.testing.assert_allclose(out_forced.state_variables[v], out_direct.state_variables[v],
                                       rtol=1e-12, atol=1e-12)
        # ...so the same diagnostics
        assert set(out_forced.diagnostic_variables) == set(out_direct.diagnostic_variables)
        for d, vals in out_direct.diagnostic_variables.items():
            np.testing.assert_allclose(np.asarray(out_forced.diagnostic_variables[d], dtype=float),
                                       np.asarray(vals, dtype=float), rtol=1e-12, atol=1e-12,
                                       err_msg=f'{name}: diagnostic {d!r} ignores the forcing on {param!r}')
        # forcing must not leak into param_values after integration
        assert np.isclose(forced.param_values[param], factory().param_values[param])


class TestCoreForcingsApplied:
    def test_restores_on_exception_t0(self):
        model = cc.Stommel()
        model.register_forcing('k', cc.Forcing(lambda t: 5.0 + 0 * t))
        with pytest.raises(RuntimeError):
            with model.forcings_applied(1.0):
                assert model.get_param_value('k', 1.0, [0, 0]) == 5.0
                raise RuntimeError
        assert model.param_values['k'] == 1.0

    def test_additive_t0(self):
        model = cc.Stommel()
        model.register_forcing('k', cc.Forcing(lambda t: 0.5 + 0 * t), attachment_style='additive')
        assert model.call_with_forcings(2.0, model.get_param_value, 'k', 2.0, [0, 0]) == 1.5
        assert model.param_values['k'] == 1.0
