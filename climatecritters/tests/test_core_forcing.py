"""Tests for climatecritters.core.forcing."""

import importlib

import numpy as np
import pytest

import climatecritters as cc
from climatecritters.signal_models import stommel


class TestForcingFromCSV:
    @pytest.mark.parametrize(
        "dataset, value_name, time_name",
        [
            ("vieira_tsi", None, None),
            ("vieira_tsi", "10", "Age (yrs BP)"),
            ("insolation", None, None),
            ("insolation", "insol_65N_d233", "kyear"),
        ],
    )
    def test_from_csv_dataset_t0(self, dataset, value_name, time_name):
        cc.Forcing.from_csv(dataset=dataset, value_name=value_name, time_name=time_name)

    @pytest.mark.parametrize("value_name", [None, "10"])
    @pytest.mark.parametrize("time_name", [None, "Age (yrs BP)"])
    def test_from_csv_file_path_t1(self, value_name, time_name):
        my_resources = importlib.resources.files("climatecritters") / "data"
        file_path = my_resources.joinpath("vieira_tsi.csv")
        cc.Forcing.from_csv(file_path=file_path, value_name=value_name, time_name=time_name)


class TestForcingCallableAndArray:
    def test_callable_with_params_t0(self):
        forcing = cc.Forcing(lambda t, amp=0.0: amp * np.asarray(t), params={"amp": 2.0})
        assert np.isclose(forcing.get_forcing(3.0), 6.0)

    @pytest.mark.parametrize("interpolation", ["linear", "cubic"])
    def test_array_interpolation_t1(self, interpolation):
        forcing = cc.Forcing(data=np.array([0.0, 1.0, 0.0]), time=np.array([0.0, 1.0, 2.0]), interpolation=interpolation)
        vals = forcing.get_forcing(np.array([0.0, 0.5, 1.0]))
        assert np.isfinite(vals).all()

    def test_array_without_time_uses_index_axis_t2(self):
        forcing = cc.Forcing(data=np.array([5.0, 6.0, 7.0]), interpolation="linear")
        assert np.isclose(forcing.get_forcing(0.0), 5.0)
        assert np.isclose(forcing.get_forcing(2.0), 7.0)

    def test_invalid_interpolation_raises_t3(self):
        with pytest.raises(ValueError, match="Unsupported interpolation"):
            cc.Forcing(data=np.array([0.0, 1.0]), interpolation="nearest")


class TestForcingSequence:
    def test_from_sequence_summary_t0(self):
        seq_forcing = cc.Forcing.from_sequence(
            [
                cc.Hold(duration=2.0, value=0.25),
                cc.Ramp(duration=3.0, y0=0.25, yf=1.0, shape="linear"),
                cc.Harmonic(duration=2.0, period=4.0, A=0.2, y0=1.0),
            ],
            label="demo",
        )

        summary = seq_forcing.summary
        assert summary is not None
        assert summary["label"] == "demo"
        assert np.isclose(summary["t_end"], 7.0)
        assert summary["n_transitions"] == 2

        vals = seq_forcing.get_forcing(np.array([0.0, 1.5, 3.0, 6.0, 8.0]))
        assert np.isfinite(vals).all()

    def test_from_elements_supports_legacy_spike_t1(self):
        forcing = cc.Forcing.from_elements(
            elements=[
                {"kind": "constant", "duration": 1.0, "value": 0.3},
                {
                    "kind": "spike",
                    "amplitude": 0.7,
                    "half_period1": 2.0,
                    "half_period2": 2.0,
                    "end_value": 0.4,
                    "shape": "cosine",
                },
            ],
            y0=0.1,
            label="legacy",
        )

        summary = forcing.summary
        assert summary is not None
        assert summary["n_parts"] == 3
        assert np.isclose(forcing.get_forcing(0.0), 0.3)
        assert np.isclose(forcing.get_forcing(summary["t_end"] + 1.0), summary["y_end"])

    def test_from_elements_invalid_kind_raises_t2(self):
        with pytest.raises(ValueError, match="Unknown forcing element kind"):
            cc.Forcing.from_elements(elements=[{"kind": "wat"}], y0=0.0)

    def test_ramp_invalid_shape_raises_t3(self):
        with pytest.raises(ValueError, match="shape"):
            cc.Ramp(duration=1.0, y0=0.0, yf=1.0, shape="sigmoid")


class TestForcingSpec:
    def _make(self, **kwargs):
        defaults = dict(
            forcing_object=cc.Forcing(lambda t: 1.0),
            attachment_style="replacement",
            timing="pre",
        )
        defaults.update(kwargs)
        from climatecritters.core.forcing import ForcingSpec
        return ForcingSpec(**defaults)

    def test_valid_replacement_pre_t0(self):
        spec = self._make(attachment_style="replacement", timing="pre")
        assert spec.attachment_style == "replacement"
        assert spec.timing == "pre"

    def test_valid_additive_post_t1(self):
        spec = self._make(attachment_style="additive", timing="post")
        assert spec.timing == "post"

    def test_invalid_attachment_style_raises_t2(self):
        with pytest.raises(ValueError, match="attachment_style"):
            self._make(attachment_style="multiply")

    def test_invalid_timing_raises_t3(self):
        with pytest.raises(ValueError, match="timing"):
            self._make(timing="during")

    def test_non_callable_forcing_object_raises_t4(self):
        with pytest.raises(TypeError, match="forcing_object"):
            self._make(forcing_object=42)

    def test_evaluate_callable_t5(self):
        from climatecritters.core.forcing import ForcingSpec
        spec = ForcingSpec(forcing_object=lambda t: t * 2.0, attachment_style="replacement", timing="pre")
        assert np.isclose(spec.evaluate(3.0), 6.0)

    def test_evaluate_forcing_object_t6(self):
        from climatecritters.core.forcing import ForcingSpec
        spec = ForcingSpec(forcing_object=cc.Forcing(lambda t: t + 1.0), attachment_style="additive", timing="post")
        assert np.isclose(spec.evaluate(4.0), 5.0)

    def test_object_with_get_forcing_method_accepted_t7(self):
        """Any object with get_forcing is valid, not just cc.Forcing."""
        from climatecritters.core.forcing import ForcingSpec

        class CustomForcing:
            def get_forcing(self, t):
                return 99.0

        spec = ForcingSpec(forcing_object=CustomForcing(), attachment_style="replacement", timing="pre")
        assert np.isclose(spec.evaluate(0.0), 99.0)


class TestForcingModelIntegration:
    def test_stommel_sequence_forcing_t0(self):
        forcing = cc.Forcing.from_sequence(
            [
                cc.Hold(duration=0.02, value=0.0),
                cc.Ramp(duration=0.03, y0=0.0, yf=0.2, shape="linear"),
            ],
            label="stommel_sequence",
        )
        model = stommel.Stommel(E=0.0)
        model.register_forcing('S', forcing, attachment_style='additive', timing='pre')
        model.integrate(t_span=(0, 0.05), y0=[1.0, 0.1], method="euler", dt=0.01)
        assert np.isfinite(model.state_variables["S"][-1])
