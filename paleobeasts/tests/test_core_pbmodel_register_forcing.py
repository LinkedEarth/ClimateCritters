"""Tests for PBModel.register_forcing, get_forcings, and clear_forcings.

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}
"""

import warnings

import numpy as np
import pytest

import paleobeasts as pb
from paleobeasts.core.forcing import ForcingSpec
from paleobeasts.signal_models import lorenz, stommel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forcing():
    """Minimal valid Forcing object for use in tests."""
    return pb.Forcing(lambda t: 1.0)


# ---------------------------------------------------------------------------
# Parameter namespace
# ---------------------------------------------------------------------------

class TestRegisterForcingParameter:
    def test_parameter_defaults_replacement_pre_t0(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('sigma', _forcing())
        specs = model.get_forcings('sigma')
        assert len(specs) == 1
        assert specs[0].attachment_style == 'replacement'
        assert specs[0].timing == 'pre'

    def test_parameter_explicit_replacement_t1(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('rho', _forcing(), attachment_style='replacement')
        specs = model.get_forcings('rho')
        assert specs[0].attachment_style == 'replacement'

    def test_parameter_additive_raises_t2(self):
        model = lorenz.Lorenz63(forcing=None)
        with pytest.raises(ValueError, match="attachment_style='replacement'"):
            model.register_forcing('sigma', _forcing(), attachment_style='additive')

    def test_parameter_timing_pre_explicit_ok_t3(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('beta', _forcing(), timing='pre')
        assert model.get_forcings('beta')[0].timing == 'pre'

    def test_parameter_timing_post_raises_t4(self):
        model = lorenz.Lorenz63(forcing=None)
        with pytest.raises(ValueError, match="pre-step"):
            model.register_forcing('sigma', _forcing(), timing='post')

    def test_parameter_double_replacement_raises_t5(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('rho', _forcing())
        with pytest.raises(ValueError, match="replacement forcing is already registered"):
            model.register_forcing('rho', _forcing())


# ---------------------------------------------------------------------------
# State variable namespace
# ---------------------------------------------------------------------------

class TestRegisterForcingState:
    def test_state_no_attachment_style_raises_t0(self):
        model = lorenz.Lorenz63(forcing=None)
        with pytest.raises(ValueError, match="attachment_style is required"):
            model.register_forcing('x', _forcing())

    def test_state_replacement_defaults_post_t1(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('x', _forcing(), attachment_style='replacement')
        specs = model.get_forcings('x')
        assert specs[0].timing == 'post'

    def test_state_replacement_pre_warns_and_uses_post_t2(self):
        model = lorenz.Lorenz63(forcing=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.register_forcing('x', _forcing(), attachment_style='replacement', timing='pre')
        assert any(issubclass(w.category, UserWarning) for w in caught)
        assert model.get_forcings('x')[0].timing == 'post'

    def test_state_additive_pre_t3(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('y', _forcing(), attachment_style='additive', timing='pre')
        spec = model.get_forcings('y')[0]
        assert spec.attachment_style == 'additive'
        assert spec.timing == 'pre'

    def test_state_additive_post_t4(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('z', _forcing(), attachment_style='additive', timing='post')
        assert model.get_forcings('z')[0].timing == 'post'

    def test_state_additive_missing_timing_raises_t5(self):
        model = lorenz.Lorenz63(forcing=None)
        with pytest.raises(ValueError, match="timing is required"):
            model.register_forcing('x', _forcing(), attachment_style='additive')

    def test_state_double_replacement_raises_t6(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('x', _forcing(), attachment_style='replacement')
        with pytest.raises(ValueError, match="replacement forcing is already registered"):
            model.register_forcing('x', _forcing(), attachment_style='replacement')

    def test_state_multiple_additive_allowed_t7(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('x', _forcing(), attachment_style='additive', timing='pre')
        model.register_forcing('x', pb.Forcing(lambda t: 2.0), attachment_style='additive', timing='pre')
        assert len(model.get_forcings('x')) == 2


# ---------------------------------------------------------------------------
# Unknown variable
# ---------------------------------------------------------------------------

class TestRegisterForcingUnknownVar:
    def test_unknown_var_raises_t0(self):
        model = lorenz.Lorenz63(forcing=None)
        with pytest.raises(ValueError, match="not found"):
            model.register_forcing('not_a_var', _forcing())

    def test_error_message_lists_valid_names_t1(self):
        model = lorenz.Lorenz63(forcing=None)
        with pytest.raises(ValueError) as exc_info:
            model.register_forcing('bogus', _forcing())
        assert 'sigma' in str(exc_info.value) or 'rho' in str(exc_info.value)


# ---------------------------------------------------------------------------
# Registry inspection and clearing
# ---------------------------------------------------------------------------

class TestGetAndClearForcings:
    def test_get_forcings_all_t0(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('sigma', _forcing())
        model.register_forcing('x', _forcing(), attachment_style='additive', timing='pre')
        registry = model.get_forcings()
        assert 'sigma' in registry
        assert 'x' in registry

    def test_get_forcings_empty_var_returns_empty_list_t1(self):
        model = lorenz.Lorenz63(forcing=None)
        assert model.get_forcings('sigma') == []

    def test_clear_specific_var_t2(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('sigma', _forcing())
        model.register_forcing('rho', _forcing())
        model.clear_forcings('sigma')
        assert model.get_forcings('sigma') == []
        assert len(model.get_forcings('rho')) == 1

    def test_clear_all_t3(self):
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('sigma', _forcing())
        model.register_forcing('rho', _forcing())
        model.clear_forcings()
        assert model.get_forcings() == {}

    def test_clear_nonexistent_var_is_silent_t4(self):
        model = lorenz.Lorenz63(forcing=None)
        model.clear_forcings('sigma')  # should not raise


# ---------------------------------------------------------------------------
# Copy behaviour
# ---------------------------------------------------------------------------

class TestRegisterForcingCopy:
    def test_copy_carries_forcings_t0(self):
        import copy
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('sigma', _forcing())
        copied = copy.copy(model)
        assert len(copied.get_forcings('sigma')) == 1

    def test_copy_forcings_are_independent_t1(self):
        """Appending to the copy's registry must not affect the original."""
        import copy
        model = lorenz.Lorenz63(forcing=None)
        model.register_forcing('sigma', _forcing())
        copied = copy.copy(model)
        copied.register_forcing('rho', _forcing())
        assert model.get_forcings('rho') == []


# ---------------------------------------------------------------------------
# ForcingSpec.evaluate is wired correctly through register_forcing
# ---------------------------------------------------------------------------

class TestForcingSpecEvaluate:
    def test_evaluate_via_spec_t0(self):
        model = lorenz.Lorenz63(forcing=None)
        f = pb.Forcing(lambda t: t * 3.0)
        model.register_forcing('sigma', f)
        spec = model.get_forcings('sigma')[0]
        assert np.isclose(spec.evaluate(2.0), 6.0)
