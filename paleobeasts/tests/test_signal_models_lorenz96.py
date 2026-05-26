''' Tests for paleobeasts.signal_models.lorenz96

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}
'''

import numpy as np
import paleobeasts as pb

from paleobeasts.signal_models.lorenz import Lorenz96


class TestSignalModelsLorenz96Integrate:
    def test_integrate_rk45_t0(self):
        model = lorenz96.Lorenz96(forcing=None, n=5, F=8.0)
        model.integrate(t_span=(0, 5), y0=[1, 1, 1, 1, 1], method='RK45')

    def test_integrate_euler_t0(self):
        model = lorenz96.Lorenz96(forcing=None, n=5, F=8.0)
        model.integrate(t_span=(0, 5), y0=[1, 1, 1, 1, 1], method='euler', kwargs={'dt': 0.01})

    def test_integrate_l96_rk4_t0(self):
        model = lorenz96.Lorenz96TwoScale(K=4, J=2, F=10.0, h=1.0, b=10.0, c=10.0, exact_rhs=True)
        y0 = np.zeros(4 + 4 * 2)
        model.integrate(t_span=(0, 1.0), y0=y0, method='l96_rk4', kwargs={'dt': 0.01, 'si': 0.05})


class TestSignalModelsLorenz96toPyleo:
    def test_topyleo_t0(self):
        model = lorenz96.Lorenz96(forcing=None, n=5, F=8.0)
        model.integrate(t_span=(0, 5), y0=[1, 1, 1, 1, 1], method='RK45')
        model.to_pyleo(var_names=['x0', 'x1'])


class TestSignalModelsLorenz96TimeVaryingParams:
    def test_time_varying_param_matches_constant_t0(self):
        model_const = lorenz96.Lorenz96(forcing=None, n=5, F=8.0)
        model_tv = lorenz96.Lorenz96(forcing=None, n=5, F=lambda t, x, m: 8.0)

        t_span = (0, 0.05)
        kwargs = {'dt': 0.01}
        y0 = [1, 1, 1, 1, 1]
        model_const.integrate(t_span=t_span, y0=y0, method='euler', kwargs=kwargs)
        model_tv.integrate(t_span=t_span, y0=y0, method='euler', kwargs=kwargs)

        const_last = np.array([model_const.state_variables[f'x{i}'][-1] for i in range(5)])
        tv_last = np.array([model_tv.state_variables[f'x{i}'][-1] for i in range(5)])

        assert np.allclose(const_last, tv_last, rtol=1e-8, atol=1e-10)


class TestSignalModelsLorenz96TwoScale:
    def test_integrate_euler_t0(self):
        K = 5
        J = 2
        model = lorenz96.Lorenz96TwoScale(K=K, J=J, F=10.0, h=1.0, b=10.0, c=10.0)
        y0 = np.zeros(K + K * J)
        model.integrate(t_span=(0, 0.05), y0=y0, method='euler', kwargs={'dt': 0.01})

        assert model.state_variables.shape[0] > 0
        assert model.state_variables.dtype.names[0] == 'x0'
        assert model.state_variables.dtype.names[K] == 'y0'
