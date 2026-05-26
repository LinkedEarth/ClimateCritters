"""Tests for paleobeasts.signal_models.pendulum."""

import numpy as np
import pytest
import paleobeasts as pb

from paleobeasts.signal_models.pendulum import SimplePendulum, DrivenPendulum, DoublePendulum


# ---------------------------------------------------------------------------
# SimplePendulum
# ---------------------------------------------------------------------------

class TestSignalModelsSimplePendulumIntegrate:
    @pytest.mark.parametrize('y0', [[0.1, 0.0], [0.5, 0.0]])
    @pytest.mark.parametrize('method, dt', [('euler', 0.01), ('RK45', None)])
    def test_integrate_t0(self, y0, method, dt):
        model = SimplePendulum(forcing=None, L=1.0, g=9.81, damping=0.0)
        model.integrate(t_span=(0, 5), y0=y0, method=method, dt=dt)

        assert model.state_variables.dtype.names == ('theta', 'omega')
        assert 'energy' in model.diagnostic_variables
        assert 'omega_0' in model.diagnostic_variables
        assert np.all(np.isfinite(model.state_variables['theta']))


class TestSignalModelsSimplePendulumsotoPyleo:
    @pytest.mark.parametrize('var_names', ['theta', 'omega', ['theta', 'omega']])
    def test_topyleo_t0(self, var_names):
        model = SimplePendulum(forcing=None)
        output = model.integrate(t_span=(0, 5), y0=[0.1, 0.0], method='RK45')
        output.to_pyleo(var_names=var_names)


class TestSignalModelsSimplePendulumPhysics:
    def test_small_angle_period_t0(self):
        """Small-amplitude oscillations should match the linear period 2π/ω₀."""
        L, g = 1.0, 9.81
        model = SimplePendulum(forcing=None, L=L, g=g, damping=0.0)
        T0 = 2.0 * np.pi * np.sqrt(L / g)

        model.integrate(
            t_span=(0, 5 * T0), y0=[0.05, 0.0], method='RK45',
            kwargs={'rtol': 1e-10, 'atol': 1e-12,
                    't_eval': np.linspace(0, 5 * T0, 5000)},
        )

        theta = model.state_variables['theta']
        time = model.time
        # find zero-crossings (θ going positive → negative)
        sign_changes = np.where(np.diff(np.sign(theta)) < 0)[0]
        assert len(sign_changes) >= 4
        # half-periods between consecutive downward crossings
        half_periods = np.diff(time[sign_changes])
        full_periods = half_periods[::2] * 2.0  # every other half-period = full period
        assert np.allclose(full_periods, T0, rtol=0.02)

    def test_undamped_energy_conservation_t1(self):
        """Zero damping, no forcing: mechanical energy must be conserved."""
        model = SimplePendulum(forcing=None, L=1.0, g=9.81, damping=0.0)
        model.integrate(
            t_span=(0, 20), y0=[0.3, 0.0], method='RK45',
            kwargs={'rtol': 1e-10, 'atol': 1e-12},
        )
        energy = model.diagnostic_variables['energy']
        assert np.max(np.abs(energy - energy[0])) < 1e-5

    def test_damped_energy_decreases_t2(self):
        """Positive damping must reduce energy over time."""
        model = SimplePendulum(forcing=None, L=1.0, g=9.81, damping=0.5)
        model.integrate(t_span=(0, 20), y0=[0.5, 0.0], method='RK45')
        energy = model.diagnostic_variables['energy']
        assert energy[-1] < energy[0]

    def test_omega_0_diagnostic_matches_helper_t3(self):
        """omega_0 diagnostic should equal natural_frequency() at every step."""
        L, g = 2.0, 9.81
        model = SimplePendulum(forcing=None, L=L, g=g, damping=0.0)
        model.integrate(t_span=(0, 5), y0=[0.1, 0.0], method='RK45')
        expected = np.sqrt(g / L)
        assert np.allclose(model.diagnostic_variables['omega_0'], expected)


class TestSignalModelsSimplePendulumConvenienceMethods:
    def test_natural_frequency_t0(self):
        model = SimplePendulum(forcing=None, L=2.0, g=9.81)
        assert np.isclose(model.natural_frequency(), np.sqrt(9.81 / 2.0))

    def test_natural_period_t1(self):
        model = SimplePendulum(forcing=None, L=1.0, g=9.81)
        assert np.isclose(model.natural_period(), 2.0 * np.pi / np.sqrt(9.81))

    def test_damping_ratio_t2(self):
        L, g, lam = 1.0, 9.81, 0.5
        model = SimplePendulum(forcing=None, L=L, g=g, damping=lam)
        expected = lam / (2.0 * np.sqrt(g / L))
        assert np.isclose(model.damping_ratio(), expected)


class TestSignalModelsSimplePendulumInvalidParams:
    def test_nonpositive_L_raises_t0(self):
        model = SimplePendulum(forcing=None, L=-1.0, g=9.81)
        with pytest.raises(ValueError, match="L must be > 0"):
            model.integrate(t_span=(0, 1), y0=[0.1, 0.0], method='euler', dt=0.01)

    def test_nonpositive_g_raises_t1(self):
        model = SimplePendulum(forcing=None, L=1.0, g=0.0)
        with pytest.raises(ValueError, match="g must be > 0"):
            model.integrate(t_span=(0, 1), y0=[0.1, 0.0], method='euler', dt=0.01)


# ---------------------------------------------------------------------------
# DrivenPendulum
# ---------------------------------------------------------------------------

class TestSignalModelsDrivenPendulumIntegrate:
    @pytest.mark.parametrize('A, dt', [(0.5, 0.01), (1.2, 0.01)])
    def test_integrate_t0(self, A, dt):
        model = DrivenPendulum(forcing=None, q=0.5, A=A, Omega=2.0 / 3.0)
        model.integrate(t_span=(0, 20), y0=[0.2, 0.0], method='euler', dt=dt)

        assert model.state_variables.dtype.names == ('theta', 'omega')
        assert 'energy' in model.diagnostic_variables
        assert 'drive' in model.diagnostic_variables

    def test_forcing_overrides_cosine_drive_t1(self):
        """Providing a Forcing object should replace the built-in cosine drive."""
        const_drive = pb.core.Forcing(lambda t: 0.5)
        model_forced = DrivenPendulum(forcing=const_drive, q=0.5)
        model_forced.integrate(t_span=(0, 5), y0=[0.0, 0.0], method='RK45')

        drive = model_forced.diagnostic_variables['drive']
        assert np.allclose(drive, 0.5)

    def test_driving_period_helper_t2(self):
        Omega = 2.0 / 3.0
        model = DrivenPendulum(forcing=None, Omega=Omega)
        assert np.isclose(model.driving_period(), 2.0 * np.pi / Omega)

    def test_topyleo_t3(self):
        model = DrivenPendulum(forcing=None)
        output = model.integrate(t_span=(0, 10), y0=[0.2, 0.0], method='RK45')
        output.to_pyleo(var_names='theta')


# ---------------------------------------------------------------------------
# DoublePendulum
# ---------------------------------------------------------------------------

class TestSignalModelsDoublePendulumIntegrate:
    def test_integrate_t0(self):
        model = DoublePendulum(forcing=None)
        model.integrate(t_span=(0, 5), y0=[np.pi / 2, 0.0, np.pi / 4, 0.0],
                        method='RK45')

        assert model.state_variables.dtype.names == ('theta1', 'omega1', 'theta2', 'omega2')
        for diag in ('energy', 'x1', 'y1', 'x2', 'y2'):
            assert diag in model.diagnostic_variables
        assert np.all(np.isfinite(model.state_variables['theta1']))

    def test_energy_conservation_t1(self):
        """No damping: total energy should be conserved to solver tolerance."""
        model = DoublePendulum(forcing=None, d1=0.0, d2=0.0)
        model.integrate(
            t_span=(0, 10), y0=[np.pi / 4, 0.0, np.pi / 4, 0.0],
            method='RK45', kwargs={'rtol': 1e-10, 'atol': 1e-12},
        )
        energy = model.diagnostic_variables['energy']
        assert np.max(np.abs(energy - energy[0])) < 1e-4

    def test_damping_reduces_energy_t2(self):
        model = DoublePendulum(forcing=None, d1=0.1, d2=0.1)
        model.integrate(t_span=(0, 20), y0=[np.pi / 4, 0.0, np.pi / 4, 0.0],
                        method='RK45')
        energy = model.diagnostic_variables['energy']
        assert energy[-1] < energy[0]

    def test_cartesian_positions_t3(self):
        """cartesian_positions() should return four finite arrays of the right length."""
        model = DoublePendulum(forcing=None)
        model.integrate(t_span=(0, 5), y0=[0.5, 0.0, 0.5, 0.0], method='RK45')
        x1, y1, x2, y2 = model.cartesian_positions()

        n = len(model.time)
        for arr in (x1, y1, x2, y2):
            assert len(arr) == n
            assert np.all(np.isfinite(arr))

    def test_topyleo_t4(self):
        model = DoublePendulum(forcing=None)
        output = model.integrate(t_span=(0, 5), y0=[0.5, 0.0, 0.5, 0.0],
                                 method='RK45')
        output.to_pyleo(var_names=['theta1', 'theta2'])
