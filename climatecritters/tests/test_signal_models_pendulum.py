"""Tests for climatecritters.model_critters.pendulum."""

import numpy as np
import pytest
import climatecritters as cc

from climatecritters.model_critters.pendulum import (
    DoublePendulum,
    DrivenPendulum,
    MultiPendulumBeta,
    PendulumRodBeta,
    SimplePendulum,
)


# ---------------------------------------------------------------------------
# SimplePendulum
# ---------------------------------------------------------------------------

class TestSignalModelsSimplePendulumIntegrate:
    @pytest.mark.parametrize('y0', [[0.1, 0.0], [0.5, 0.0]])
    @pytest.mark.parametrize('method, dt', [('euler', 0.01), ('RK45', None)])
    def test_integrate_t0(self, y0, method, dt):
        model = SimplePendulum(L=1.0, g=9.81, damping=0.0)
        model.integrate(t_span=(0, 5), y0=y0, method=method, dt=dt)

        assert model.state_variables.dtype.names == ('theta', 'omega')
        assert 'energy' in model.diagnostic_variables
        assert 'omega_0' in model.diagnostic_variables
        assert np.all(np.isfinite(model.state_variables['theta']))


class TestSignalModelsSimplePendulumsotoPyleo:
    @pytest.mark.parametrize('var_names', ['theta', 'omega', ['theta', 'omega']])
    def test_topyleo_t0(self, var_names):
        model = SimplePendulum()
        output = model.integrate(t_span=(0, 5), y0=[0.1, 0.0], method='RK45')
        output.to_pyleo(var_names=var_names)


class TestSignalModelsSimplePendulumPhysics:
    def test_small_angle_period_t0(self):
        """Small-amplitude oscillations should match the linear period 2π/ω₀."""
        L, g = 1.0, 9.81
        model = SimplePendulum(L=L, g=g, damping=0.0)
        T0 = 2.0 * np.pi * np.sqrt(L / g)

        model.integrate(
            t_span=(0, 5 * T0), y0=[0.05, 0.0], method='RK45',
            kwargs={'rtol': 1e-6, 'atol': 1e-8,
                    't_eval': np.linspace(0, 5 * T0, 5000)},
        )

        theta = model.state_variables['theta']
        time = model.time
        # consecutive downward zero-crossings are exactly one full period apart
        sign_changes = np.where(np.diff(np.sign(theta)) < 0)[0]
        assert len(sign_changes) >= 4
        periods = np.diff(time[sign_changes])
        assert np.allclose(periods, T0, rtol=0.02)

    def test_undamped_energy_conservation_t1(self):
        """Zero damping, no forcing: mechanical energy must be conserved."""
        model = SimplePendulum(L=1.0, g=9.81, damping=0.0)
        model.integrate(
            t_span=(0, 20), y0=[0.3, 0.0], method='RK45',
            kwargs={'rtol': 1e-6, 'atol': 1e-8},
        )
        energy = model.diagnostic_variables['energy']
        assert np.max(np.abs(energy - energy[0])) < 1e-5

    def test_damped_energy_decreases_t2(self):
        """Positive damping must reduce energy over time."""
        model = SimplePendulum(L=1.0, g=9.81, damping=0.5)
        model.integrate(t_span=(0, 20), y0=[0.5, 0.0], method='RK45')
        energy = model.diagnostic_variables['energy']
        assert energy[-1] < energy[0]

    def test_omega_0_diagnostic_matches_helper_t3(self):
        """omega_0 diagnostic should equal natural_frequency() at every step."""
        L, g = 2.0, 9.81
        model = SimplePendulum(L=L, g=g, damping=0.0)
        model.integrate(t_span=(0, 5), y0=[0.1, 0.0], method='RK45')
        expected = np.sqrt(g / L)
        assert np.allclose(model.diagnostic_variables['omega_0'], expected)


class TestSignalModelsSimplePendulumConvenienceMethods:
    def test_natural_frequency_t0(self):
        model = SimplePendulum(L=2.0, g=9.81)
        assert np.isclose(model.natural_frequency(), np.sqrt(9.81 / 2.0))

    def test_natural_period_t1(self):
        model = SimplePendulum(L=1.0, g=9.81)
        assert np.isclose(model.natural_period(), 2.0 * np.pi / np.sqrt(9.81))

    def test_damping_ratio_t2(self):
        L, g, lam = 1.0, 9.81, 0.5
        model = SimplePendulum(L=L, g=g, damping=lam)
        expected = lam / (2.0 * np.sqrt(g / L))
        assert np.isclose(model.damping_ratio(), expected)


class TestSignalModelsSimplePendulumInvalidParams:
    def test_nonpositive_L_raises_t0(self):
        model = SimplePendulum(L=-1.0, g=9.81)
        with pytest.raises(ValueError, match="L must be > 0"):
            model.integrate(t_span=(0, 1), y0=[0.1, 0.0], method='euler', dt=0.01)

    def test_nonpositive_g_raises_t1(self):
        model = SimplePendulum(L=1.0, g=0.0)
        with pytest.raises(ValueError, match="g must be > 0"):
            model.integrate(t_span=(0, 1), y0=[0.1, 0.0], method='euler', dt=0.01)


# ---------------------------------------------------------------------------
# DrivenPendulum
# ---------------------------------------------------------------------------

class TestSignalModelsDrivenPendulumIntegrate:
    @pytest.mark.parametrize('A, dt', [(0.5, 0.01), (1.2, 0.01)])
    def test_integrate_t0(self, A, dt):
        model = DrivenPendulum(q=0.5, A=A, Omega=2.0 / 3.0)
        model.integrate(t_span=(0, 20), y0=[0.2, 0.0], method='euler', dt=dt)

        assert model.state_variables.dtype.names == ('theta', 'omega')
        assert 'energy' in model.diagnostic_variables
        assert 'drive' in model.diagnostic_variables

    def test_zero_amplitude_gives_zero_drive_t1(self):
        """A=0 produces zero drive at every step."""
        model = DrivenPendulum(q=0.5, A=0.0)
        model.integrate(t_span=(0, 5), y0=[0.0, 0.0], method='RK45')

        drive = model.diagnostic_variables['drive']
        assert np.allclose(drive, 0.0)

    def test_driving_period_helper_t2(self):
        Omega = 2.0 / 3.0
        model = DrivenPendulum(Omega=Omega)
        assert np.isclose(model.driving_period(), 2.0 * np.pi / Omega)

    def test_topyleo_t3(self):
        model = DrivenPendulum()
        output = model.integrate(t_span=(0, 10), y0=[0.2, 0.0], method='RK45')
        output.to_pyleo(var_names='theta')


# ---------------------------------------------------------------------------
# DoublePendulum
# ---------------------------------------------------------------------------

class TestSignalModelsDoublePendulumIntegrate:
    def test_integrate_t0(self):
        model = DoublePendulum()
        model.integrate(t_span=(0, 5), y0=[np.pi / 2, 0.0, np.pi / 4, 0.0],
                        method='RK45')

        assert model.state_variables.dtype.names == ('theta1', 'omega1', 'theta2', 'omega2')
        for diag in ('energy', 'x1', 'y1', 'x2', 'y2'):
            assert diag in model.diagnostic_variables
        assert np.all(np.isfinite(model.state_variables['theta1']))

    def test_energy_conservation_t1(self):
        """No damping: total energy should be conserved to solver tolerance."""
        model = DoublePendulum(d1=0.0, d2=0.0)
        model.integrate(
            t_span=(0, 10), y0=[np.pi / 4, 0.0, np.pi / 4, 0.0],
            method='RK45', kwargs={'rtol': 1e-6, 'atol': 1e-8},
        )
        energy = model.diagnostic_variables['energy']
        assert np.max(np.abs(energy - energy[0])) < 1e-4

    def test_damping_reduces_energy_t2(self):
        model = DoublePendulum(d1=0.1, d2=0.1)
        model.integrate(t_span=(0, 20), y0=[np.pi / 4, 0.0, np.pi / 4, 0.0],
                        method='RK45')
        energy = model.diagnostic_variables['energy']
        assert energy[-1] < energy[0]

    def test_cartesian_positions_t3(self):
        """cartesian_positions() should return four finite arrays of the right length."""
        model = DoublePendulum()
        model.integrate(t_span=(0, 5), y0=[0.5, 0.0, 0.5, 0.0], method='RK45')
        x1, y1, x2, y2 = model.cartesian_positions()

        n = len(model.time)
        for arr in (x1, y1, x2, y2):
            assert len(arr) == n
            assert np.all(np.isfinite(arr))

    def test_topyleo_t4(self):
        model = DoublePendulum()
        output = model.integrate(t_span=(0, 5), y0=[0.5, 0.0, 0.5, 0.0],
                                 method='RK45')
        output.to_pyleo(var_names=['theta1', 'theta2'])


# ---------------------------------------------------------------------------
# MultiPendulumBeta
# ---------------------------------------------------------------------------

class TestSignalModelsMultiPendulumBetaIntegrate:
    def test_integrate_t0(self):
        rods = [
            PendulumRodBeta(m=1.0, L=1.0),
            PendulumRodBeta(m=1.0, L=1.0),
            PendulumRodBeta(m=0.5, L=0.8),
        ]
        model = MultiPendulumBeta(rods=rods)
        model.integrate(
            t_span=(0, 2),
            y0=[0.3, 0.0, 0.2, 0.0, 0.1, 0.0],
            method='RK45',
        )

        assert model.state_variables.dtype.names == (
            'theta1', 'omega1', 'theta2', 'omega2', 'theta3', 'omega3'
        )
        for diag in ('energy', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'):
            assert diag in model.diagnostic_variables
        assert np.all(np.isfinite(model.state_variables['theta1']))

    def test_matches_double_pendulum_for_two_rods_t1(self):
        y0 = [0.4, 0.0, 0.2, 0.0]
        t_eval = np.linspace(0.0, 1.0, 201)
        kwargs = {'rtol': 1e-6, 'atol': 1e-8, 't_eval': t_eval}

        double = DoublePendulum(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81)
        multi = MultiPendulumBeta(
            rods=[PendulumRodBeta(m=1.0, L=1.0), PendulumRodBeta(m=1.0, L=1.0)],
            g=9.81,
        )

        double.integrate(t_span=(0, 1), y0=y0, method='RK45', kwargs=kwargs)
        multi.integrate(t_span=(0, 1), y0=y0, method='RK45', kwargs=kwargs)

        assert np.allclose(double.time, multi.time)
        assert np.allclose(double.state_variables['theta1'], multi.state_variables['theta1'])
        assert np.allclose(double.state_variables['omega1'], multi.state_variables['omega1'])
        assert np.allclose(double.state_variables['theta2'], multi.state_variables['theta2'])
        assert np.allclose(double.state_variables['omega2'], multi.state_variables['omega2'])

    def test_callable_rod_forcing_t2(self):
        rods = [
            PendulumRodBeta(m=1.0, L=1.0, forcing=lambda t: 0.1 * np.cos(t)),
            PendulumRodBeta(m=1.0, L=1.0),
        ]
        model = MultiPendulumBeta(rods=rods)
        model.integrate(t_span=(0, 1), y0=[0.1, 0.0, 0.1, 0.0], method='RK45')

        assert np.all(np.isfinite(model.diagnostic_variables['energy']))

    def test_cartesian_positions_t3(self):
        rods = [PendulumRodBeta(m=1.0, L=1.0), PendulumRodBeta(m=1.0, L=1.0)]
        model = MultiPendulumBeta(rods=rods)
        model.integrate(t_span=(0, 1), y0=[0.2, 0.0, 0.1, 0.0], method='RK45')

        positions = model.cartesian_positions()
        assert len(positions) == 4
        for arr in positions:
            assert len(arr) == len(model.time)
            assert np.all(np.isfinite(arr))


class TestSignalModelsMultiPendulumBetaInvalidParams:
    def test_requires_at_least_two_rods_t0(self):
        with pytest.raises(ValueError, match="at least 2 rods"):
            MultiPendulumBeta(rods=[PendulumRodBeta(m=1.0, L=1.0)])

    def test_nonpositive_g_raises_t1(self):
        rods = [PendulumRodBeta(m=1.0, L=1.0), PendulumRodBeta(m=1.0, L=1.0)]
        with pytest.raises(ValueError, match="g must be > 0"):
            MultiPendulumBeta(rods=rods, g=0.0)

    def test_invalid_rod_raises_t2(self):
        with pytest.raises(ValueError, match="length must be > 0"):
            PendulumRodBeta(m=1.0, L=0.0)
