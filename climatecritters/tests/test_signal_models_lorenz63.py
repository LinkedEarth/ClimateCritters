''' Tests for climatecritters.model_critters.lorenz (Lorenz63)

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

from climatecritters.model_critters import lorenz


class TestSignalModelsLorenz63Integrate:
    @pytest.mark.parametrize('y0', [[1, 1, 1], [0, 1, 2]])
    @pytest.mark.parametrize('t_span', [(0, 10), (0, 20)])
    @pytest.mark.parametrize('method, dt', [('euler', 0.01), ('RK45', None)])
    def test_integrate_t0(self, t_span, y0, method, dt):
        '''Test integrate method'''
        model = lorenz.Lorenz63()
        model.integrate(t_span=t_span, y0=y0, method=method, dt=dt)


class TestSignalModelsLorenz63toPyleo:
    @pytest.mark.parametrize('method, dt', [('euler', 0.01), ('RK45', None)])
    @pytest.mark.parametrize('var_names', ['x', 'y', 'z', ['x', 'y'], ['x', 'y', 'z']])
    def test_topyleo_t0(self, method, dt, var_names):
        '''Test to_pyleo method on CCOutput'''
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


class TestSignalModelsLorenz63PostHistory:
    '''Output is built from the solver trajectory, not accumulated in dydt (#51).'''

    def test_dydt_pure_t0(self):
        '''dydt is callable before integrate and has no side effects'''
        model = lorenz.Lorenz63(sigma=10.0, rho=28.0, beta=8 / 3)
        rhs = model.dydt(0.0, [1.0, 2.0, 3.0])
        assert np.allclose(rhs, [10.0, 1.0 * (28.0 - 3.0) - 2.0, 2.0 - 8.0])
        model.integrate(t_span=(0, 1), y0=[1, 1, 1])
        n = len(model.state_variables)
        model.dydt(0.5, [1.0, 2.0, 3.0])
        assert len(model.state_variables) == n

    def test_rk45_matches_accepted_states_t0(self):
        '''RK45 output equals solve_ivp's accepted states, not stage evaluations'''
        from scipy.integrate import solve_ivp

        def rhs(t, s):
            x, y, z = s
            return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]

        ref = solve_ivp(rhs, (0, 5), [1.0, 1.0, 1.0], method='RK45', rtol=1e-8, atol=1e-10)
        out = lorenz.Lorenz63().integrate(t_span=(0, 5), y0=[1, 1, 1], method='RK45')
        got = np.column_stack([out.state_variables[v] for v in 'xyz'])
        assert np.allclose(out.time, ref.t)
        assert np.allclose(got, ref.y.T, rtol=0, atol=1e-12)

    def test_euler_includes_final_time_t0(self):
        out = lorenz.Lorenz63().integrate(t_span=(0, 1), y0=[1, 1, 1], method='euler', dt=0.01)
        assert len(out.time) == 101
        assert np.isclose(out.time[-1], 1.0)
        assert len(out.state_variables) == len(out.time)

    def test_rk4_t0(self):
        '''rk4 (with si subsampling) runs and agrees with RK45 over a short span'''
        out_rk4 = lorenz.Lorenz63().integrate(t_span=(0, 1), y0=[1, 1, 1], method='rk4',
                                              dt=0.001, kwargs={'si': 0.01})
        assert len(out_rk4.time) == 101
        out_ref = lorenz.Lorenz63().integrate(t_span=(0, 1), y0=[1, 1, 1], method='RK45',
                                              output_time=out_rk4.time)
        for v in 'xyz':
            assert np.allclose(out_rk4.state_variables[v], out_ref.state_variables[v],
                               rtol=1e-5, atol=1e-6)
