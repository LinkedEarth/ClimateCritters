"""Tests for paleobeasts.signal_models.daisyworld."""

import numpy as np
import pytest
import paleobeasts as pb

from paleobeasts.signal_models import daisyworld


class TestSignalModelsDaisyworldIntegrate:
    @pytest.mark.parametrize('y0', [[0.2, 0.2, 288.0], [0.1, 0.3, 290.0]])
    @pytest.mark.parametrize('t_span', [(0, 2), (0, 5)])
    @pytest.mark.parametrize('method, dt', [('euler', 0.01), ('RK45', None)])
    def test_integrate_t0(self, t_span, y0, method, dt):
        model = daisyworld.Daisyworld()
        model.integrate(t_span=t_span, y0=y0, method=method, dt=dt)

        assert model.state_variables.dtype.names == ('Aw', 'Ab', 'T')
        assert 'A_planet' in model.diagnostic_variables
        assert 'beta_w' in model.diagnostic_variables


class TestSignalModelsDaisyworldTimeVaryingParams:
    def test_time_varying_params_match_constants_t0(self):
        model_const = daisyworld.Daisyworld(
            alpha_w=0.75, alpha_b=0.25, gamma=0.3, L=1.0, C=10.0
        )
        model_tv = daisyworld.Daisyworld(
            alpha_w=lambda t, x, m: 0.75,
            alpha_b=lambda t: 0.25,
            gamma=lambda t, x: 0.3,
            L=lambda t, x: 1.0,
            C=lambda t: 10.0,
        )

        t_span = (0, 0.05)
        y0 = [0.2, 0.2, 288.0]
        model_const.integrate(t_span=t_span, y0=y0, method='euler', dt=0.01)
        model_tv.integrate(t_span=t_span, y0=y0, method='euler', dt=0.01)

        const_last = np.array([
            model_const.state_variables['Aw'][-1],
            model_const.state_variables['Ab'][-1],
            model_const.state_variables['T'][-1],
        ])
        tv_last = np.array([
            model_tv.state_variables['Aw'][-1],
            model_tv.state_variables['Ab'][-1],
            model_tv.state_variables['T'][-1],
        ])
        assert np.allclose(const_last, tv_last, rtol=1e-8, atol=1e-10)


class TestSignalModelsDaisyworldForcing:
    def test_luminosity_forcing_changes_temperature_t0(self):
        forced = daisyworld.Daisyworld()
        forced.register_forcing('L', pb.core.Forcing(lambda t: 0.1))
        unforced = daisyworld.Daisyworld()
        t_span = (0, 0.05)
        y0 = [0.2, 0.2, 288.0]

        forced.integrate(t_span=t_span, y0=y0, method='euler', dt=0.01)
        unforced.integrate(t_span=t_span, y0=y0, method='euler', dt=0.01)

        assert not np.isclose(forced.state_variables['T'][-1], unforced.state_variables['T'][-1])
