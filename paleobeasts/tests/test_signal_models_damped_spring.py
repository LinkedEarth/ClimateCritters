"""Tests for paleobeasts.signal_models.damped_spring."""

import numpy as np
import pytest
import paleobeasts as pb

from paleobeasts.signal_models.damped_spring import DampedSpring


class TestSignalModelsDampedSpringIntegrate:
    @pytest.mark.parametrize('y0', [[1.0, 0.0], [0.0, 1.0]])
    @pytest.mark.parametrize('method, dt', [('euler', 0.01), ('RK45', None)])
    def test_integrate_t0(self, y0, method, dt):
        model = DampedSpring(m=1.0, k=1.0, c=0.1)
        model.integrate(t_span=(0, 10), y0=y0, method=method, dt=dt)

        assert model.state_variables.dtype.names == ('x', 'v')
        assert 'energy' in model.diagnostic_variables
        assert 'omega_0' in model.diagnostic_variables
        assert np.all(np.isfinite(model.state_variables['x']))
        assert np.all(np.isfinite(model.state_variables['v']))


class TestSignalModelsDampedSpringtoPyleo:
    @pytest.mark.parametrize('var_names', ['x', 'v', ['x', 'v']])
    def test_topyleo_t0(self, var_names):
        model = DampedSpring()
        output = model.integrate(t_span=(0, 5), y0=[1.0, 0.0], method='RK45')
        output.to_pyleo(var_names=var_names)


class TestSignalModelsDampedSpringPhysics:
    def test_undamped_energy_conservation_t0(self):
        """c=0, no forcing: total mechanical energy should be conserved."""
        model = DampedSpring(m=1.0, k=4.0, c=0.0)
        model.integrate(
            t_span=(0, 20), y0=[1.0, 0.0], method='RK45',
            kwargs={'rtol': 1e-10, 'atol': 1e-12},
        )
        energy = model.diagnostic_variables['energy']
        assert np.max(np.abs(energy - energy[0])) < 1e-6

    def test_damped_energy_decreases_t1(self):
        """Positive damping must dissipate energy over time."""
        model = DampedSpring(m=1.0, k=1.0, c=0.5)
        model.integrate(t_span=(0, 20), y0=[1.0, 0.0], method='RK45')
        energy = model.diagnostic_variables['energy']
        assert energy[-1] < energy[0]

    def test_driven_at_resonance_grows_amplitude_t2(self):
        """Undamped oscillator driven at ω₀ should grow in amplitude."""
        m, k = 1.0, 4.0
        omega_0 = np.sqrt(k / m)
        forcing = pb.core.Forcing(lambda t: np.cos(omega_0 * t))
        model = DampedSpring(m=m, k=k, c=0.0)
        model.register_forcing('F', forcing)
        model.integrate(t_span=(0, 30), y0=[0.0, 0.0], method='RK45')

        x = model.state_variables['x']
        # amplitude should grow; last quarter should exceed first quarter
        n = len(x)
        assert np.max(np.abs(x[3 * n // 4:])) > np.max(np.abs(x[:n // 4]))

    def test_overdamped_no_oscillation_t3(self):
        """ζ > 1 (overdamped): displacement should return to zero monotonically."""
        # c = 2*sqrt(k*m)*2 gives ζ = 2 (overdamped)
        m, k = 1.0, 1.0
        c = 4.0 * np.sqrt(k * m)
        model = DampedSpring(m=m, k=k, c=c)
        model.integrate(t_span=(0, 20), y0=[1.0, 0.0], method='RK45')

        x = model.state_variables['x']
        assert x[-1] < x[0]
        assert np.all(x >= 0.0)  # no oscillation through zero


class TestSignalModelsDampedSpringConvenienceMethods:
    def test_natural_frequency_t0(self):
        model = DampedSpring(m=2.0, k=8.0, c=0.0)
        assert np.isclose(model.natural_frequency(), np.sqrt(8.0 / 2.0))

    def test_natural_period_t1(self):
        model = DampedSpring(m=1.0, k=4.0, c=0.0)
        assert np.isclose(model.natural_period(), 2.0 * np.pi / 2.0)

    def test_damping_ratio_t2(self):
        m, k, c = 1.0, 4.0, 2.0
        model = DampedSpring(m=m, k=k, c=c)
        expected = c / (2.0 * np.sqrt(k * m))
        assert np.isclose(model.damping_ratio(), expected)


class TestSignalModelsDampedSpringTimeVaryingParams:
    def test_time_varying_params_match_constants_t0(self):
        model_const = DampedSpring(m=1.0, k=2.0, c=0.2)
        model_tv = DampedSpring(
            m=lambda t: 1.0,
            k=lambda t, x: 2.0,
            c=lambda t, x, m: 0.2,
        )

        y0 = [1.0, 0.0]
        model_const.integrate(t_span=(0, 1), y0=y0, method='euler', dt=0.01)
        model_tv.integrate(t_span=(0, 1), y0=y0, method='euler', dt=0.01)

        const_last = np.array([model_const.state_variables['x'][-1],
                                model_const.state_variables['v'][-1]])
        tv_last = np.array([model_tv.state_variables['x'][-1],
                             model_tv.state_variables['v'][-1]])
        assert np.allclose(const_last, tv_last, rtol=1e-8, atol=1e-10)


class TestSignalModelsDampedSpringInvalidParams:
    def test_nonpositive_mass_raises_t0(self):
        model = DampedSpring(m=-1.0, k=1.0, c=0.0)
        with pytest.raises(ValueError, match="m must be > 0"):
            model.integrate(t_span=(0, 1), y0=[1.0, 0.0], method='euler', dt=0.1)

    def test_nonpositive_spring_constant_raises_t1(self):
        model = DampedSpring(m=1.0, k=0.0, c=0.0)
        with pytest.raises(ValueError, match="k must be > 0"):
            model.integrate(t_span=(0, 1), y0=[1.0, 0.0], method='euler', dt=0.1)
