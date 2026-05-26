"""Tests for paleobeasts.core.forcing."""

import importlib

import numpy as np
import pytest

import paleobeasts as pb
from paleobeasts.signal_models import stommel


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
        pb.Forcing.from_csv(dataset=dataset, value_name=value_name, time_name=time_name)

    @pytest.mark.parametrize("value_name", [None, "10"])
    @pytest.mark.parametrize("time_name", [None, "Age (yrs BP)"])
    def test_from_csv_file_path_t1(self, value_name, time_name):
        my_resources = importlib.resources.files("paleobeasts") / "data"
        file_path = my_resources.joinpath("vieira_tsi.csv")
        pb.Forcing.from_csv(file_path=file_path, value_name=value_name, time_name=time_name)


class TestForcingCallableAndArray:
    def test_callable_with_params_t0(self):
        forcing = pb.Forcing(lambda t, amp=0.0: amp * np.asarray(t), params={"amp": 2.0})
        assert np.isclose(forcing.get_forcing(3.0), 6.0)

    @pytest.mark.parametrize("interpolation", ["linear", "cubic"])
    def test_array_interpolation_t1(self, interpolation):
        forcing = pb.Forcing(data=np.array([0.0, 1.0, 0.0]), time=np.array([0.0, 1.0, 2.0]), interpolation=interpolation)
        vals = forcing.get_forcing(np.array([0.0, 0.5, 1.0]))
        assert np.isfinite(vals).all()

    def test_array_without_time_uses_index_axis_t2(self):
        forcing = pb.Forcing(data=np.array([5.0, 6.0, 7.0]), interpolation="linear")
        assert np.isclose(forcing.get_forcing(0.0), 5.0)
        assert np.isclose(forcing.get_forcing(2.0), 7.0)

    def test_invalid_interpolation_raises_t3(self):
        with pytest.raises(ValueError, match="Unsupported interpolation"):
            pb.Forcing(data=np.array([0.0, 1.0]), interpolation="nearest")


class TestForcingSequence:
    def test_from_sequence_summary_t0(self):
        seq_forcing = pb.Forcing.from_sequence(
            [
                pb.Hold(duration=2.0, value=0.25),
                pb.Ramp(duration=3.0, y0=0.25, yf=1.0, shape="linear"),
                pb.Harmonic(duration=2.0, period=4.0, A=0.2, y0=1.0),
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
        forcing = pb.Forcing.from_elements(
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
            pb.Forcing.from_elements(elements=[{"kind": "wat"}], y0=0.0)

    def test_ramp_invalid_shape_raises_t3(self):
        with pytest.raises(ValueError, match="shape"):
            pb.Ramp(duration=1.0, y0=0.0, yf=1.0, shape="sigmoid")


class TestForcingModelIntegration:
    def test_stommel_sequence_forcing_t0(self):
        forcing = pb.Forcing.from_sequence(
            [
                pb.Hold(duration=0.02, value=0.0),
                pb.Ramp(duration=0.03, y0=0.0, yf=0.2, shape="linear"),
            ],
            label="stommel_sequence",
        )
        model = stommel.Stommel(forcing=forcing, E=0.0)
        model.integrate(t_span=(0, 0.05), y0=[1.0, 0.1], method="euler", dt=0.01)
        assert np.isfinite(model.state_variables["S"][-1])
