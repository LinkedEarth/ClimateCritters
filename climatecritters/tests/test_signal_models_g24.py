''' Tests for climatecritters.model_critters.g24

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

from climatecritters.model_critters import g24

class TestSignalModelsG24Integrate:
    @pytest.mark.parametrize('y0', [[0,1],[1,1]])
    @pytest.mark.parametrize('t_span', [(0,10),(0,100)])
    @pytest.mark.parametrize('method, dt', [('euler', 1), ('RK45', None)])
    def test_integrate_t0(self,t_span,y0,method,dt):
        '''Test integrate method'''
        def func(x):
            return 1
        forcing = cc.Forcing(func)
        model3 = g24.Model3()
        model3.register_forcing('insolation', forcing)
        model3.integrate(t_span=t_span,y0=y0,method=method,dt=dt)

class TestSignalModelsG24toPyleo:
    @pytest.mark.parametrize('method, dt', [('euler', 1), ('RK45', None)])
    @pytest.mark.parametrize('var_names', ['v','k','insolation', ['v','k'], ['k','v','insolation']])
    def test_topyleo_t0(self,method,dt,var_names):
        '''Test to_pyleo method'''
        def func(x):
            return 1
        forcing = cc.Forcing(func)
        model3 = g24.Model3()
        model3.register_forcing('insolation', forcing)
        output = model3.integrate(t_span=(0,10),y0=[1,1],method=method,dt=dt)
        output.to_pyleo(var_names=var_names)


class TestSignalModelsG24TimeVaryingParams:
    def test_time_varying_params_match_constants_t0(self):
        forcing = cc.Forcing(lambda t: 1.0)

        model_const = g24.Model3(f1=-16, f2=16, t1=30, t2=10, vc=1.4)
        model_const.register_forcing('insolation', forcing)
        model_tv = g24.Model3(
            f1=lambda t: -16,
            f2=lambda t, x: 16,
            t1=lambda t, x: 30,
            t2=lambda t, x: 10,
            vc=lambda t, x, m: 1.4,
        )
        model_tv.register_forcing('insolation', forcing)

        t_span = (0, 10)
        y0 = [1, 1]
        model_const.integrate(t_span=t_span, y0=y0, method='euler', dt=1)
        model_tv.integrate(t_span=t_span, y0=y0, method='euler', dt=1)

        const_last = np.array([model_const.state_variables['v'][-1], model_const.state_variables['k'][-1]])
        tv_last = np.array([model_tv.state_variables['v'][-1], model_tv.state_variables['k'][-1]])

        assert np.allclose(const_last, tv_last, rtol=1e-8, atol=1e-10)

    def test_time_varying_params_strict_contract_t1(self):
        forcing = cc.Forcing(lambda t: 1.0)

        model_tv = g24.Model3(
            f1=lambda t: -16,
            f2=lambda t, x: 16,
            t1=lambda t, x, m: 30,
            t2=lambda t: 10,
            vc=lambda t, x: 1.4,
        )
        model_tv.register_forcing('insolation', forcing)

        model_tv.integrate(t_span=(0, 10), y0=[1, 1], method='euler', dt=1)
        assert np.isfinite(model_tv.state_variables['v'][-1])

    def test_dfdt_attribute_assignment_updates_param_store_t2(self):
        model3 = g24.Model3()
        model3.dfdt = lambda t: 123.0
        assert model3.param_values['dfdt'](0.0) == 123.0
        assert model3.calc_dfdt(0.0, [1.0]) == 123.0


class TestSignalModelsG24SequenceForcing:
    def test_sequence_forcing_integrates_t0(self):
        forcing = cc.Forcing.from_sequence(
            [
                cc.forcing.Hold(duration=4.0, value=1.0),
                cc.forcing.Harmonic(duration=6.0, period=4.0, A=0.2, y0=1.0),
            ],
            label='g24_sequence',
        )
        model3 = g24.Model3()
        model3.register_forcing('insolation', forcing)
        model3.integrate(t_span=(0, 10), y0=[1, 1], method='euler', dt=1)
        assert np.isfinite(model3.state_variables['v'][-1])
