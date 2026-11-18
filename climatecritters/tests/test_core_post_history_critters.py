''' Regression tests for critters that build their output after integration (#51, #57)

These critters used to append to ``state_variables`` inside ``dydt``, which
recorded intermediate solver stages instead of accepted states (RK45), dropped
the final time point (euler), and blocked ``method='rk4'``.

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}
'''

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import climatecritters as cc
from climatecritters.model_critters.ebm import EBM0D

# (id, factory, y0, t_end)
CRITTERS = [
    ('Lorenz63', cc.Lorenz63, [1.0, 1.0, 1.0], 5.0),
    ('Roessler', cc.Roessler, [1.0, 1.0, 1.0], 20.0),
    ('Daisyworld', cc.Daisyworld, [0.2, 0.2, 295.0], 100.0),
    ('EBM0D', EBM0D, [288.0], 50.0),
]
IDS = [c[0] for c in CRITTERS]


def _states(out):
    return np.column_stack([out.state_variables[v] for v in out.state_variable_names])


class TestCorePostHistoryCritters:
    @pytest.mark.parametrize('name, factory, y0, t_end', CRITTERS, ids=IDS)
    def test_dydt_pure_t0(self, name, factory, y0, t_end):
        '''dydt is callable before integrate and leaves the model untouched'''
        model = factory()
        assert model.uses_post_history is True
        first = np.asarray(model.dydt(0.0, y0), dtype=float)
        second = np.asarray(model.dydt(0.0, y0), dtype=float)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize('name, factory, y0, t_end', CRITTERS, ids=IDS)
    def test_rk45_matches_accepted_states_t0(self, name, factory, y0, t_end):
        '''RK45 output equals an independent solve_ivp run on the same dydt'''
        out = factory().integrate(t_span=(0, t_end), y0=y0, method='RK45')
        ref = solve_ivp(factory().dydt, (0, t_end), y0, method='RK45', rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(out.time, ref.t)
        np.testing.assert_allclose(_states(out), ref.y.T, rtol=0, atol=1e-12)

    @pytest.mark.parametrize('name, factory, y0, t_end', CRITTERS, ids=IDS)
    def test_euler_includes_final_time_t0(self, name, factory, y0, t_end):
        t_stop = t_end / 10
        dt = t_stop / 100           # small enough for explicit Euler to stay stable
        out = factory().integrate(t_span=(0, t_stop), y0=y0, method='euler', dt=dt)
        assert len(out.time) == 101
        assert np.isclose(out.time[-1], t_stop)
        assert len(out.state_variables) == len(out.time)
        assert np.all(np.isfinite(_states(out)))
        for vals in out.diagnostic_variables.values():
            assert len(vals) == len(out.time)

    @pytest.mark.parametrize('name, factory, y0, t_end', CRITTERS, ids=IDS)
    def test_rk4_t0(self, name, factory, y0, t_end):
        '''rk4 with si subsampling runs and agrees with RK45'''
        # fine dt: Daisyworld's clipping (max(0, .)) degrades RK4's convergence order
        dt, si = t_end / 20000, t_end / 100
        out = factory().integrate(t_span=(0, t_end / 10), y0=y0, method='rk4',
                                  dt=dt, kwargs={'si': si})
        ref = factory().integrate(t_span=(0, t_end / 10), y0=y0, method='RK45',
                                  output_time=out.time)
        np.testing.assert_allclose(_states(out), _states(ref), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize('name, factory, y0, t_end',
                             [c for c in CRITTERS if c[0] in ('Daisyworld', 'EBM0D')],
                             ids=['Daisyworld', 'EBM0D'])
    def test_diagnostics_at_accepted_states_t0(self, name, factory, y0, t_end):
        '''Diagnostics are evaluated at the accepted states'''
        model = factory()
        out = model.integrate(t_span=(0, t_end), y0=y0, method='RK45')
        states = _states(out)
        names = list(out.diagnostic_variables)
        for i in (0, len(out.time) // 2, -1):
            expected = model._evaluate(out.time[i], states[i])[1]
            got = [out.diagnostic_variables[n][i] for n in names]
            np.testing.assert_allclose(got, expected, rtol=1e-12)
