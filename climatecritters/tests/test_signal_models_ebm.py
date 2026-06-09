''' Tests for climatecritters.model_critters.ebm

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import pytest
import numpy as np
import climatecritters as cc

from climatecritters.model_critters import ebm


class TestSignalModelsEBM0DIntegrate:
    @pytest.mark.parametrize('y0', [[1], [10]])
    @pytest.mark.parametrize('t_span', [(0, 10), (0, 100)])
    @pytest.mark.parametrize('OLR', [None, ebm.OLR_func(1000, 1000)])
    @pytest.mark.parametrize('method, dt', [('euler', 1), ('RK45', None)])
    def test_integrate_t0(self, t_span, y0, method, OLR, dt):
        '''Test integrate method'''
        forcing = cc.core.Forcing(lambda x: 1)
        model = ebm.EBM0D(OLR=OLR)
        model.register_forcing('S0', forcing)
        model.integrate(t_span=t_span, y0=y0, method=method, dt=dt)


class TestSignalModelsEBM0DtoPyleo:
    @pytest.mark.parametrize('method, dt', [('euler', 1), ('RK45', None)])
    @pytest.mark.parametrize('var_names', [
        'T',
        'albedo',
        'absorbed_SW',
        'OLR',
        'solar_incoming',
        ['T', 'albedo'],
        ['T', 'albedo', 'absorbed_SW', 'OLR', 'solar_incoming'],
    ])
    def test_topyleo_t0(self, method, dt, var_names):
        '''Test to_pyleo method'''
        forcing = cc.core.Forcing(lambda x: 1)
        model = ebm.EBM0D()
        model.register_forcing('S0', forcing)
        output = model.integrate(t_span=(0, 10), y0=[100], method=method, dt=dt)
        output.to_pyleo(var_names=var_names)


class TestSignalModelsEBM0DTimeVaryingParams:
    def test_time_varying_params_match_constants_t0(self):
        forcing = cc.core.Forcing(lambda t: 1360.0)

        model_const = ebm.EBM0D(C=4.0, albedo=0.3)
        model_const.register_forcing('S0', forcing)
        model_tv = ebm.EBM0D(
            C=lambda t, state, model: 4.0,
            albedo=lambda t, state: 0.3,
        )
        model_tv.register_forcing('S0', forcing)

        t_span = (0, 10)
        output_const = model_const.integrate(t_span=t_span, y0=[280], method='euler', dt=1)
        output_tv = model_tv.integrate(t_span=t_span, y0=[280], method='euler', dt=1)

        const_last = output_const.state_variables['T'][-1]
        tv_last = output_tv.state_variables['T'][-1]

        assert np.isclose(const_last, tv_last, rtol=1e-8, atol=1e-10)


class TestSignalModelsEBM0DSequenceForcing:
    def test_sequence_forcing_integrates_t0(self):
        forcing = cc.core.Forcing.from_sequence(
            [
                cc.core.Hold(duration=6.0, value=1360.0),
                cc.core.Ramp(duration=4.0, y0=1360.0, yf=1365.0, shape='linear'),
            ],
            label='ebm_sequence',
        )
        model = ebm.EBM0D()
        model.register_forcing('S0', forcing)
        output = model.integrate(t_span=(0, 10), y0=[280], method='euler', dt=1)
        assert len(output.time) > 1
        assert np.isfinite(output.state_variables['T'][-1])


class TestSignalModelsEBM1DLatSmoke:
    def test_ebm1dlat_integrates_t0(self):
        '''EBM1DLat is importable, integrates, and produces expected diagnostics.'''
        model = ebm.EBM1DLat(S0=1365.0)
        output = model.integrate(t_span=(0, 50), y0=[15.0])
        assert len(output.diagnostic_variables['Tglobal']) == len(output.time)
        assert np.isfinite(output.diagnostic_variables['Tglobal'][-1])
