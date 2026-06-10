"""Tests for CCModel.register_forcing, get_forcings, and clear_forcings.

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}
"""

import warnings

import numpy as np
import pytest

import climatecritters as cc
from climatecritters.core.forcing import ForcingSpec
from climatecritters.model_critters import lorenz, stommel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forcing():
    """Minimal valid Forcing object for use in tests."""
    return cc.Forcing(lambda t: 1.0)


# ---------------------------------------------------------------------------
# Parameter namespace
# ---------------------------------------------------------------------------

class TestRegisterForcingParameter:
    def test_parameter_defaults_replacement_pre_t0(self):
        model = lorenz.Lorenz63()
        model.register_forcing('sigma', _forcing())
        specs = model.get_forcings('sigma')
        assert len(specs) == 1
        assert specs[0].attachment_style == 'replacement'
        assert specs[0].timing == 'pre'

    def test_parameter_explicit_replacement_t1(self):
        model = lorenz.Lorenz63()
        model.register_forcing('rho', _forcing(), attachment_style='replacement')
        specs = model.get_forcings('rho')
        assert specs[0].attachment_style == 'replacement'

    def test_parameter_additive_allowed_t2(self):
        """additive is now valid for parameters (e.g. noise centred on a constant)."""
        model = lorenz.Lorenz63()
        model.register_forcing('sigma', _forcing(), attachment_style='additive')
        specs = model.get_forcings('sigma')
        assert len(specs) == 1
        assert specs[0].attachment_style == 'additive'
        assert specs[0].timing == 'pre'

    def test_parameter_timing_pre_explicit_ok_t3(self):
        model = lorenz.Lorenz63()
        model.register_forcing('beta', _forcing(), timing='pre')
        assert model.get_forcings('beta')[0].timing == 'pre'

    def test_parameter_timing_post_raises_t4(self):
        model = lorenz.Lorenz63()
        with pytest.raises(ValueError, match="pre-step"):
            model.register_forcing('sigma', _forcing(), timing='post')

    def test_parameter_double_replacement_raises_t5(self):
        model = lorenz.Lorenz63()
        model.register_forcing('rho', _forcing())
        with pytest.raises(ValueError, match="replacement forcing is already registered"):
            model.register_forcing('rho', _forcing())


# ---------------------------------------------------------------------------
# State variable namespace
# ---------------------------------------------------------------------------

class TestRegisterForcingState:
    def test_state_no_attachment_style_raises_t0(self):
        model = lorenz.Lorenz63()
        with pytest.raises(ValueError, match="attachment_style is required"):
            model.register_forcing('x', _forcing())

    def test_state_replacement_defaults_post_t1(self):
        model = lorenz.Lorenz63()
        model.register_forcing('x', _forcing(), attachment_style='replacement')
        specs = model.get_forcings('x')
        assert specs[0].timing == 'post'

    def test_state_replacement_pre_warns_and_uses_post_t2(self):
        model = lorenz.Lorenz63()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.register_forcing('x', _forcing(), attachment_style='replacement', timing='pre')
        assert any(issubclass(w.category, UserWarning) for w in caught)
        assert model.get_forcings('x')[0].timing == 'post'

    def test_state_additive_pre_t3(self):
        model = lorenz.Lorenz63()
        model.register_forcing('y', _forcing(), attachment_style='additive', timing='pre')
        spec = model.get_forcings('y')[0]
        assert spec.attachment_style == 'additive'
        assert spec.timing == 'pre'

    def test_state_additive_post_t4(self):
        model = lorenz.Lorenz63()
        model.register_forcing('z', _forcing(), attachment_style='additive', timing='post')
        assert model.get_forcings('z')[0].timing == 'post'

    def test_state_additive_missing_timing_raises_t5(self):
        model = lorenz.Lorenz63()
        with pytest.raises(ValueError, match="timing is required"):
            model.register_forcing('x', _forcing(), attachment_style='additive')

    def test_state_double_replacement_raises_t6(self):
        model = lorenz.Lorenz63()
        model.register_forcing('x', _forcing(), attachment_style='replacement')
        with pytest.raises(ValueError, match="replacement forcing is already registered"):
            model.register_forcing('x', _forcing(), attachment_style='replacement')

    def test_state_multiple_additive_allowed_t7(self):
        model = lorenz.Lorenz63()
        model.register_forcing('x', _forcing(), attachment_style='additive', timing='pre')
        model.register_forcing('x', cc.Forcing(lambda t: 2.0), attachment_style='additive', timing='pre')
        assert len(model.get_forcings('x')) == 2


# ---------------------------------------------------------------------------
# Unknown variable
# ---------------------------------------------------------------------------

class TestRegisterForcingUnknownVar:
    def test_unknown_var_raises_t0(self):
        model = lorenz.Lorenz63()
        with pytest.raises(ValueError, match="not found"):
            model.register_forcing('not_a_var', _forcing())

    def test_error_message_lists_valid_names_t1(self):
        model = lorenz.Lorenz63()
        with pytest.raises(ValueError) as exc_info:
            model.register_forcing('bogus', _forcing())
        assert 'sigma' in str(exc_info.value) or 'rho' in str(exc_info.value)


# ---------------------------------------------------------------------------
# Registry inspection and clearing
# ---------------------------------------------------------------------------

class TestGetAndClearForcings:
    def test_get_forcings_all_t0(self):
        model = lorenz.Lorenz63()
        model.register_forcing('sigma', _forcing())
        model.register_forcing('x', _forcing(), attachment_style='additive', timing='pre')
        registry = model.get_forcings()
        assert 'sigma' in registry
        assert 'x' in registry

    def test_get_forcings_empty_var_returns_empty_list_t1(self):
        model = lorenz.Lorenz63()
        assert model.get_forcings('sigma') == []

    def test_clear_specific_var_t2(self):
        model = lorenz.Lorenz63()
        model.register_forcing('sigma', _forcing())
        model.register_forcing('rho', _forcing())
        model.clear_forcings('sigma')
        assert model.get_forcings('sigma') == []
        assert len(model.get_forcings('rho')) == 1

    def test_clear_all_t3(self):
        model = lorenz.Lorenz63()
        model.register_forcing('sigma', _forcing())
        model.register_forcing('rho', _forcing())
        model.clear_forcings()
        assert model.get_forcings() == {}

    def test_clear_nonexistent_var_is_silent_t4(self):
        model = lorenz.Lorenz63()
        model.clear_forcings('sigma')  # should not raise


# ---------------------------------------------------------------------------
# Copy behaviour
# ---------------------------------------------------------------------------

class TestRegisterForcingCopy:
    def test_copy_carries_forcings_t0(self):
        import copy
        model = lorenz.Lorenz63()
        model.register_forcing('sigma', _forcing())
        copied = copy.copy(model)
        assert len(copied.get_forcings('sigma')) == 1

    def test_copy_forcings_are_independent_t1(self):
        """Appending to the copy's registry must not affect the original."""
        import copy
        model = lorenz.Lorenz63()
        model.register_forcing('sigma', _forcing())
        copied = copy.copy(model)
        copied.register_forcing('rho', _forcing())
        assert model.get_forcings('rho') == []


# ---------------------------------------------------------------------------
# ForcingSpec.evaluate is wired correctly through register_forcing
# ---------------------------------------------------------------------------

class TestForcingSpecEvaluate:
    def test_evaluate_via_spec_t0(self):
        model = lorenz.Lorenz63()
        f = cc.Forcing(lambda t: t * 3.0)
        model.register_forcing('sigma', f)
        spec = model.get_forcings('sigma')[0]
        assert np.isclose(spec.evaluate(2.0), 6.0)


# ---------------------------------------------------------------------------
# Integration behaviour — pre-step parameter replacement
# ---------------------------------------------------------------------------

class TestIntegrationPreStepParameter:
    def test_parameter_replacement_affects_trajectory_t0(self):
        """A constant parameter forcing should produce the same trajectory as
        setting that parameter value directly."""
        f_const = 12.0

        # baseline: set sigma directly
        m_base = lorenz.Lorenz63()
        m_base.sigma = f_const
        out_base = m_base.integrate(t_span=(0, 1), y0=[1, 1, 1], method='euler', dt=0.01)

        # forced: register a constant forcing on sigma
        m_forced = lorenz.Lorenz63()
        m_forced.register_forcing('sigma', cc.Forcing(lambda t: f_const))
        out_forced = m_forced.integrate(t_span=(0, 1), y0=[1, 1, 1], method='euler', dt=0.01)

        np.testing.assert_allclose(
            out_forced.state_variables['x'],
            out_base.state_variables['x'],
            rtol=1e-10,
        )

    def test_parameter_replacement_restores_after_step_t1(self):
        """param_values must be restored to original after each dydt call."""
        model = lorenz.Lorenz63()
        original_sigma = model.sigma
        model.register_forcing('sigma', cc.Forcing(lambda t: original_sigma * 2))
        model.integrate(t_span=(0, 0.1), y0=[1, 1, 1], method='euler', dt=0.01)
        assert model.param_values['sigma'] == original_sigma

    def test_parameter_additive_shifts_trajectory_t2(self):
        """Additive parameter forcing should give same result as replacement with
        nominal + perturbation as the forced value."""
        sigma_0 = 10.0
        delta = 2.0

        m_additive = lorenz.Lorenz63(sigma=sigma_0)
        m_additive.register_forcing('sigma', cc.Forcing(lambda t: delta),
                                    attachment_style='additive')

        m_replace = lorenz.Lorenz63(sigma=sigma_0)
        m_replace.register_forcing('sigma', cc.Forcing(lambda t: sigma_0 + delta))

        y0 = [1.0, 1.0, 1.0]
        out_add = m_additive.integrate(t_span=(0, 1), y0=y0, method='euler', dt=0.01)
        out_rep = m_replace.integrate(t_span=(0, 1), y0=y0, method='euler', dt=0.01)

        np.testing.assert_allclose(
            out_add.state_variables['x'],
            out_rep.state_variables['x'],
            rtol=1e-10,
        )

    def test_parameter_additive_restores_after_step_t3(self):
        """param_values must be restored to the original nominal value after each step."""
        model = lorenz.Lorenz63(sigma=10.0)
        model.register_forcing('sigma', cc.Forcing(lambda t: 3.0), attachment_style='additive')
        model.integrate(t_span=(0, 0.1), y0=[1, 1, 1], method='euler', dt=0.01)
        assert model.param_values['sigma'] == 10.0


# ---------------------------------------------------------------------------
# Integration behaviour — pre-step state additive
# ---------------------------------------------------------------------------

class TestIntegrationPreStepStateAdditive:
    def test_additive_pre_shifts_trajectory_t0(self):
        """Adding a positive constant to dx/dt should increase x relative to
        the unforced trajectory."""
        m_base = lorenz.Lorenz63()
        out_base = m_base.integrate(t_span=(0, 2), y0=[1, 1, 1], method='euler', dt=0.01)

        m_forced = lorenz.Lorenz63()
        m_forced.register_forcing('x', cc.Forcing(lambda t: 1.0),
                                  attachment_style='additive', timing='pre')
        out_forced = m_forced.integrate(t_span=(0, 2), y0=[1, 1, 1], method='euler', dt=0.01)

        # forced x trajectory should differ from baseline
        assert not np.allclose(out_forced.state_variables['x'],
                               out_base.state_variables['x'])

    def test_zero_additive_pre_matches_baseline_t1(self):
        """A zero additive forcing should leave the trajectory unchanged."""
        m_base = lorenz.Lorenz63()
        out_base = m_base.integrate(t_span=(0, 1), y0=[1, 1, 1], method='euler', dt=0.01)

        m_forced = lorenz.Lorenz63()
        m_forced.register_forcing('x', cc.Forcing(lambda t: 0.0),
                                  attachment_style='additive', timing='pre')
        out_forced = m_forced.integrate(t_span=(0, 1), y0=[1, 1, 1], method='euler', dt=0.01)

        np.testing.assert_allclose(
            out_forced.state_variables['x'],
            out_base.state_variables['x'],
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Integration behaviour — post-step state replacement and additive
# ---------------------------------------------------------------------------

class TestIntegrationPostStep:
    def test_replacement_post_pins_state_variable_t0(self):
        """Replacing y post-step should pin it to the forced value at every
        saved timestep."""
        pin_value = 5.0
        model = lorenz.Lorenz63()
        model.register_forcing('y', cc.Forcing(lambda t: pin_value),
                               attachment_style='replacement')
        out = model.integrate(t_span=(0, 2), y0=[1, 1, 1], method='euler', dt=0.01)
        # t=0 is the initial condition (post-step hasn't fired yet); skip it
        np.testing.assert_allclose(out.state_variables['y'][1:], pin_value, rtol=1e-10)

    def test_additive_post_shifts_state_t1(self):
        """Adding a constant post-step should accumulate a drift in the
        affected variable relative to the unforced run."""
        m_base = lorenz.Lorenz63()
        out_base = m_base.integrate(t_span=(0, 2), y0=[1, 1, 1], method='euler', dt=0.01)

        m_forced = lorenz.Lorenz63()
        m_forced.register_forcing('z', cc.Forcing(lambda t: 0.5),
                                  attachment_style='additive', timing='post')
        out_forced = m_forced.integrate(t_span=(0, 2), y0=[1, 1, 1], method='euler', dt=0.01)

        assert not np.allclose(out_forced.state_variables['z'],
                               out_base.state_variables['z'])

    def test_post_step_adaptive_solver_warns_t2(self):
        """Post-step forcings with an adaptive solver should emit a UserWarning."""
        model = lorenz.Lorenz63()
        model.register_forcing('y', cc.Forcing(lambda t: 0.0),
                               attachment_style='replacement')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.integrate(t_span=(0, 1), y0=[1, 1, 1], method='RK45')
        assert any(issubclass(w.category, UserWarning) and 'adaptive' in str(w.message).lower()
                   for w in caught)
