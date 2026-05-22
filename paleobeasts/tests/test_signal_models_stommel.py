"""Tests for paleobeasts.signal_models.stommel."""

import numpy as np
import pytest
import paleobeasts as pb

from paleobeasts.signal_models import stommel


class TestSignalModelsStommelIntegrate:
    @pytest.mark.parametrize('y0', [[1.0, 0.5], [0.2, -0.1]])
    @pytest.mark.parametrize('t_span', [(0, 2), (0, 5)])
    @pytest.mark.parametrize('method, kwargs', [('euler', {'dt': 0.01}), ('RK45', None)])
    def test_integrate_t0(self, t_span, y0, method, kwargs):
        model = stommel.Stommel(forcing=None)
        model.integrate(t_span=t_span, y0=y0, method=method, kwargs=kwargs)
        assert model.state_variables.dtype.names == ('T', 'S')
        assert 'q' in model.diagnostic_variables


class TestSignalModelsStommelTimeVaryingParams:
    def test_time_varying_params_match_constants_t0(self):
        forcing = pb.core.Forcing(lambda t: 0.0)

        model_const = stommel.Stommel(
            forcing=forcing,
            alpha=1.0,
            beta=1.0,
            k=1.0,
            E=0.05,
            lambda_T=1.0,
            lambda_S=1.0,
            T_star=1.0,
            S_star=0.0,
        )
        model_tv = stommel.Stommel(
            forcing=forcing,
            alpha=lambda t, x, m: 1.0,
            beta=lambda t: 1.0,
            k=lambda t, x: 1.0,
            E=lambda t, x: 0.05,
            lambda_T=lambda t: 1.0,
            lambda_S=lambda t, x: 1.0,
            T_star=lambda t: 1.0,
            S_star=lambda t: 0.0,
        )

        t_span = (0, 0.05)
        kwargs = {'dt': 0.01}
        y0 = [1.0, 0.5]

        model_const.integrate(t_span=t_span, y0=y0, method='euler', kwargs=kwargs)
        model_tv.integrate(t_span=t_span, y0=y0, method='euler', kwargs=kwargs)

        const_last = np.array([model_const.state_variables['T'][-1], model_const.state_variables['S'][-1]])
        tv_last = np.array([model_tv.state_variables['T'][-1], model_tv.state_variables['S'][-1]])
        assert np.allclose(const_last, tv_last, rtol=1e-8, atol=1e-10)


class TestSignalModelsStommelForcing:
    def test_scalar_forcing_affects_salinity_tendency_t0(self):
        forced = stommel.Stommel(forcing=pb.core.Forcing(lambda t: 0.1), E=0.0)
        unforced = stommel.Stommel(forcing=None, E=0.0)
        t_span = (0, 0.05)
        y0 = [1.0, 0.1]
        kwargs = {'dt': 0.01}

        forced.integrate(t_span=t_span, y0=y0, method='euler', kwargs=kwargs)
        unforced.integrate(t_span=t_span, y0=y0, method='euler', kwargs=kwargs)

        assert not np.isclose(forced.state_variables['S'][-1], unforced.state_variables['S'][-1])


class TestSignalModelsStommelSequenceForcing:
    def test_sequence_forcing_integrates_t0(self):
        forcing = pb.core.Forcing.from_sequence(
            [
                pb.core.Hold(duration=0.02, value=0.0),
                pb.core.Ramp(duration=0.03, y0=0.0, yf=0.1, shape='linear'),
            ],
            label='stommel_sequence',
        )
        model = stommel.Stommel(forcing=forcing, E=0.0)
        model.integrate(t_span=(0, 0.05), y0=[1.0, 0.1], method='euler', kwargs={'dt': 0.01})
        assert np.isfinite(model.state_variables['S'][-1])
