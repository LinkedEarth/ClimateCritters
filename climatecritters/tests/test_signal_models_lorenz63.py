''' Tests for climatecritters.signal_models.lorenz (Lorenz63)

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

import numpy as np
import pytest
import climatecritters as cc

from climatecritters.signal_models import lorenz


class TestSignalModelsLorenz63Integrate:
    @pytest.mark.parametrize('y0', [[1, 1, 1], [0, 1, 2]])
    @pytest.mark.parametrize('t_span', [(0, 10), (0, 100)])
    @pytest.mark.parametrize('method, dt', [('euler', 0.01), ('RK45', None)])
    def test_integrate_t0(self, t_span, y0, method, dt):
        '''Test integrate method'''
        model = lorenz.Lorenz63()
        model.integrate(t_span=t_span, y0=y0, method=method, dt=dt)


class TestSignalModelsLorenz63toPyleo:
    @pytest.mark.parametrize('method, dt', [('euler', 0.01), ('RK45', None)])
    @pytest.mark.parametrize('var_names', ['x', 'y', 'z', ['x', 'y'], ['x', 'y', 'z']])
    def test_topyleo_t0(self, method, dt, var_names):
        '''Test to_pyleo method on PBOutput'''
        model = lorenz.Lorenz63()
        output = model.integrate(t_span=(0, 10), y0=[1, 1, 1], method=method, dt=dt)
        output.to_pyleo(var_names=var_names)


class TestSignalModelsLorenz63TimeVaryingParams:
    def test_time_varying_params_match_constants_t0(self):
        model_const = lorenz.Lorenz63(sigma=10.0, rho=28.0, beta=8 / 3)
        model_tv = lorenz.Lorenz63(
            sigma=lambda t, x, m: 10.0,
            rho=lambda t: 28.0,
            beta=lambda t, x: 8 / 3,
        )

        t_span = (0, 0.05)
        model_const.integrate(t_span=t_span, y0=[1, 1, 1], method='euler', dt=0.01)
        model_tv.integrate(t_span=t_span, y0=[1, 1, 1], method='euler', dt=0.01)

        const_last = np.array([
            model_const.state_variables['x'][-1],
            model_const.state_variables['y'][-1],
            model_const.state_variables['z'][-1],
        ])
        tv_last = np.array([
            model_tv.state_variables['x'][-1],
            model_tv.state_variables['y'][-1],
            model_tv.state_variables['z'][-1],
        ])

        assert np.allclose(const_last, tv_last, rtol=1e-8, atol=1e-10)
