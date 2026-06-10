"""Tests for climatecritters.core.forcing."""

import importlib

import numpy as np
import pytest

import climatecritters as cc
from climatecritters.model_critters import stommel


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

    def test_ramp_invalid_shape_raises_t1(self):
        with pytest.raises(ValueError, match="shape"):
            cc.Ramp(duration=1.0, y0=0.0, yf=1.0, shape="sigmoid")


class TestForcingAddOperator:
    """Tests for the redesigned + operator."""

    # --- Forcing + Forcing: value superposition ---

    def test_forcing_plus_forcing_sums_values_t0(self):
        f1 = cc.Forcing(lambda t: 2.0)
        f2 = cc.Forcing(lambda t: 3.0)
        combined = f1 + f2
        assert isinstance(combined, cc.Forcing)
        assert np.isclose(combined.get_forcing(0.0), 5.0)
        assert np.isclose(combined.get_forcing(99.0), 5.0)

    def test_forcing_plus_callable_sums_values_t1(self):
        f = cc.Forcing(lambda t: 1.0)
        combined = f + (lambda t: t)
        assert np.isclose(combined.get_forcing(3.0), 4.0)

    def test_callable_plus_forcing_radd_t2(self):
        f = cc.Forcing(lambda t: 10.0)
        combined = (lambda t: t) + f
        assert np.isclose(combined.get_forcing(5.0), 15.0)

    # --- Bounded + Forcing: additive overlay, auto-compiles ---

    def test_element_plus_forcing_returns_forcing_t3(self):
        elem = cc.Hold(duration=5.0, value=2.0)
        noise = cc.Forcing(lambda t: 1.0)
        result = elem + noise
        assert isinstance(result, cc.Forcing)

    def test_element_plus_forcing_sums_within_duration_t4(self):
        elem = cc.Hold(duration=5.0, value=2.0)
        noise = cc.Forcing(lambda t: 1.0)
        result = elem + noise
        # Within duration: Hold(2) + noise(1) = 3
        assert np.isclose(result.get_forcing(2.5), 3.0)

    def test_forcing_plus_element_commutative_t5(self):
        elem = cc.Hold(duration=5.0, value=2.0)
        noise = cc.Forcing(lambda t: 1.0)
        assert np.isclose((elem + noise).get_forcing(2.5),
                          (noise + elem).get_forcing(2.5))

    def test_sequence_plus_forcing_returns_forcing_t6(self):
        seq = cc.Hold(duration=2.0, value=0.0) + cc.Hold(duration=3.0, value=1.0)
        noise = cc.Forcing(lambda t: 0.5)
        result = seq + noise
        assert isinstance(result, cc.Forcing)

    def test_sequence_plus_forcing_sums_values_t7(self):
        seq = cc.Hold(duration=5.0, value=2.0) + cc.Hold(duration=5.0, value=4.0)
        offset = cc.Forcing(lambda t: 10.0)
        result = seq + offset
        assert np.isclose(result.get_forcing(2.5), 12.0)   # Hold(2) + 10
        assert np.isclose(result.get_forcing(7.5), 14.0)   # Hold(4) + 10

    def test_forcing_plus_sequence_radd_t8(self):
        seq = cc.Hold(duration=5.0, value=3.0) + cc.Hold(duration=5.0, value=6.0)
        offset = cc.Forcing(lambda t: 1.0)
        result = offset + seq
        assert np.isclose(result.get_forcing(2.5), 4.0)
        assert np.isclose(result.get_forcing(7.5), 7.0)

    # --- Element + Element: temporal concatenation (unchanged) ---

    def test_element_plus_element_is_sequence_t9(self):
        result = cc.Hold(duration=2.0, value=0.0) + cc.Hold(duration=3.0, value=1.0)
        assert isinstance(result, cc.ForcingSequence)
        assert len(result.parts) == 2

    def test_sequence_plus_element_is_sequence_t10(self):
        seq = cc.Hold(duration=2.0, value=0.0) + cc.Hold(duration=3.0, value=1.0)
        result = seq + cc.Hold(duration=2.0, value=2.0)
        assert isinstance(result, cc.ForcingSequence)
        assert len(result.parts) == 3


class TestForcingUtils:
    """Tests for create_forcing, named factories, and make_forcing_element."""

    def test_create_forcing_no_duration_returns_forcing_t0(self):
        from climatecritters.utils.forcing import create_forcing
        f = create_forcing(lambda t: t * 2.0)
        assert isinstance(f, cc.Forcing)
        assert np.isclose(f.get_forcing(3.0), 6.0)

    def test_create_forcing_with_duration_returns_element_t1(self):
        from climatecritters.utils.forcing import create_forcing
        elem = create_forcing(lambda t: t * 2.0, duration=10.0)
        assert isinstance(elem, cc.ForcingElement)

    def test_create_constant_no_duration_t2(self):
        from climatecritters.utils.forcing import create_constant_forcing
        f = create_constant_forcing(5.0)
        assert isinstance(f, cc.Forcing)
        assert np.isclose(f.get_forcing(999.0), 5.0)

    def test_create_constant_with_duration_t3(self):
        from climatecritters.utils.forcing import create_constant_forcing
        elem = create_constant_forcing(5.0, duration=10.0)
        assert isinstance(elem, cc.ForcingElement)

    def test_create_sinusoid_no_duration_t4(self):
        from climatecritters.utils.forcing import create_sinusoid_forcing
        f = create_sinusoid_forcing(A=1.0, period=2.0 * np.pi)
        assert isinstance(f, cc.Forcing)
        assert np.isclose(f.get_forcing(0.0), 0.0, atol=1e-10)

    def test_create_sinusoid_with_duration_t5(self):
        from climatecritters.utils.forcing import create_sinusoid_forcing
        elem = create_sinusoid_forcing(A=1.0, period=1.0, duration=5.0)
        assert isinstance(elem, cc.ForcingElement)

    def test_create_periodic_single_component_matches_sinusoid_t6(self):
        from climatecritters.utils.forcing import create_sinusoid_forcing, create_periodic_forcing
        f_sin = create_sinusoid_forcing(A=2.0, period=4.0)
        f_per = create_periodic_forcing([(4.0, 1.0)], desired_amplitude=2.0)
        t_vals = np.linspace(0, 8, 50)
        np.testing.assert_allclose(
            [f_sin.get_forcing(t) for t in t_vals],
            [f_per.get_forcing(t) for t in t_vals],
            atol=1e-12,
        )

    def test_make_forcing_element_callable_backed_t7(self):
        from climatecritters.utils.forcing import make_forcing_element
        f = cc.Forcing(lambda t: np.sin(t))
        elem = make_forcing_element(f, duration=10.0)
        assert isinstance(elem, cc.ForcingElement)
        assert np.isclose(elem._duration, 10.0)

    def test_make_forcing_element_infers_duration_from_time_axis_t8(self):
        from climatecritters.utils.forcing import make_forcing_element
        f = cc.Forcing(data=np.array([0.0, 1.0, 0.0]),
                       time=np.array([0.0, 5.0, 10.0]),
                       interpolation="linear")
        elem = make_forcing_element(f)
        assert np.isclose(elem._duration, 10.0)

    def test_make_forcing_element_no_time_axis_no_duration_raises_t9(self):
        from climatecritters.utils.forcing import make_forcing_element
        f = cc.Forcing(lambda t: t)
        with pytest.raises(ValueError, match="duration must be provided"):
            make_forcing_element(f)

    def test_make_forcing_element_composable_in_sequence_t10(self):
        from climatecritters.utils.forcing import make_forcing_element
        f = cc.Forcing(lambda t: np.sin(t))
        elem = make_forcing_element(f, duration=np.pi)
        seq = cc.Hold(duration=1.0, value=0.0) + elem
        compiled = seq.compile()
        assert isinstance(compiled, cc.Forcing)
        assert np.isclose(compiled.summary["t_end"], 1.0 + np.pi)


class TestForcingElementBase:
    """Tests for ForcingElement used directly as a callable-backed segment."""

    def test_basic_construction_t0(self):
        elem = cc.ForcingElement(lambda t: t * 2.0, duration=5.0)
        assert hasattr(elem, "_func")
        assert hasattr(elem, "_duration")
        assert np.isclose(elem._duration, 5.0)

    def test_non_callable_raises_t1(self):
        with pytest.raises(TypeError, match="callable"):
            cc.ForcingElement(func=42.0, duration=5.0)

    def test_zero_duration_raises_t2(self):
        with pytest.raises(ValueError, match="duration"):
            cc.ForcingElement(lambda t: t, duration=0.0)

    def test_negative_duration_raises_t3(self):
        with pytest.raises(ValueError, match="duration"):
            cc.ForcingElement(lambda t: t, duration=-1.0)

    def test_resolve_produces_correct_segment_t4(self):
        elem = cc.ForcingElement(lambda t: t * 2.0, duration=5.0)
        seg = elem._resolve(t0=3.0, y_prev=0.0)
        assert seg.kind == "func"
        assert seg.eval_mode == "func"
        assert np.isclose(seg.t0, 3.0)
        assert np.isclose(seg.tf, 8.0)
        assert np.isclose(seg.y0, 6.0)   # func(3.0) = 6.0
        assert np.isclose(seg.yf, 16.0)  # func(8.0) = 16.0

    def test_in_sequence_evaluates_correctly_t5(self):
        """ForcingElement embedded in a ForcingSequence evaluates func(t) correctly."""
        # sin element starting at t=2, duration=4 — absolute t passed to func
        elem = cc.ForcingElement(lambda t: np.sin(t), duration=4.0)
        seq_forcing = cc.Forcing.from_sequence(
            [cc.Hold(duration=2.0, value=0.0), elem],
            label="test",
        )
        # At t=3 (tau=1 into the elem, absolute t=3): expect sin(3)
        assert np.isclose(seq_forcing.get_forcing(3.0), np.sin(3.0), atol=1e-10)
        # At t=5 (tau=3, absolute t=5): expect sin(5)
        assert np.isclose(seq_forcing.get_forcing(5.0), np.sin(5.0), atol=1e-10)

    def test_concatenation_with_named_elements_t6(self):
        """ForcingElement composes with Hold/Ramp via + operator."""
        elem = cc.ForcingElement(lambda t: 1.0, duration=3.0)
        seq = cc.Hold(duration=2.0, value=0.0) + elem
        assert isinstance(seq, cc.ForcingSequence)
        assert len(seq.parts) == 2

    def test_sequence_duration_includes_func_element_t7(self):
        elem = cc.ForcingElement(lambda t: 0.5, duration=3.0)
        seq_forcing = cc.Forcing.from_sequence(
            [cc.Hold(duration=2.0, value=0.0), elem],
        )
        assert np.isclose(seq_forcing.summary["t_end"], 5.0)


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


class TestPlotKwargs:
    """Tests for plot_kwargs propagation and .plot() return values."""

    # --- plot_kwargs stored on all element types ---

    def test_hold_stores_plot_kwargs_t0(self):
        elem = cc.Hold(duration=5.0, value=1.0, plot_kwargs={'color': 'red'})
        assert elem.plot_kwargs == {'color': 'red'}

    def test_ramp_stores_plot_kwargs_t1(self):
        elem = cc.Ramp(duration=5.0, y0=0.0, yf=1.0, plot_kwargs={'color': 'blue'})
        assert elem.plot_kwargs == {'color': 'blue'}

    def test_harmonic_stores_plot_kwargs_t2(self):
        elem = cc.Harmonic(duration=5.0, period=1.0, A=0.5, y0=0.0,
                           plot_kwargs={'linestyle': '--'})
        assert elem.plot_kwargs == {'linestyle': '--'}

    def test_forcing_element_stores_plot_kwargs_t3(self):
        elem = cc.ForcingElement(lambda t: t, duration=5.0, plot_kwargs={'alpha': 0.5})
        assert elem.plot_kwargs == {'alpha': 0.5}

    def test_default_plot_kwargs_is_none_t4(self):
        assert cc.Hold(duration=5.0, value=1.0).plot_kwargs is None
        assert cc.Ramp(duration=5.0, y0=0.0, yf=1.0).plot_kwargs is None
        assert cc.Harmonic(duration=5.0, period=1.0, A=0.5, y0=0.0).plot_kwargs is None

    # --- .plot() return types ---

    def test_forcing_element_plot_returns_fig_ax_t5(self):
        import matplotlib
        matplotlib.use('Agg')
        elem = cc.Hold(duration=10.0, value=2.0)
        fig, ax = elem.plot()
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_forcing_sequence_plot_returns_fig_ax_t6(self):
        import matplotlib
        matplotlib.use('Agg')
        seq = cc.Hold(10.0, value=0.0) + cc.Ramp(10.0, y0=0.0, yf=1.0) + cc.Hold(10.0, value=1.0)
        fig, ax = seq.plot()
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_forcing_plot_sequence_backed_infers_t_span_t7(self):
        import matplotlib
        matplotlib.use('Agg')
        f = cc.Forcing.from_sequence([cc.Hold(10.0, value=1.0), cc.Hold(10.0, value=2.0)])
        fig, ax = f.plot()
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_forcing_plot_callable_no_t_span_raises_t8(self):
        f = cc.Forcing(lambda t: t)
        with pytest.raises(ValueError, match="t_span is required"):
            f.plot()

    def test_forcing_plot_callable_with_t_span_t9(self):
        import matplotlib
        matplotlib.use('Agg')
        f = cc.Forcing(lambda t: np.sin(t))
        fig, ax = f.plot(t_span=(0, 10))
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_forcing_plot_array_backed_infers_t_span_t10(self):
        import matplotlib
        matplotlib.use('Agg')
        f = cc.Forcing(data=np.array([0.0, 1.0, 0.0]),
                       time=np.array([0.0, 5.0, 10.0]),
                       interpolation='linear')
        fig, ax = f.plot()
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_sequence_plot_uses_element_plot_kwargs_t11(self):
        """The line drawn for a Hold with plot_kwargs color='red' should be red."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        seq = cc.Hold(10.0, value=1.0, plot_kwargs={'color': 'red'})
        fig, ax = seq.plot()
        line_color = ax.lines[0].get_color()
        assert line_color == 'red'
        plt.close(fig)

    def test_sequence_plot_accepts_existing_ax_t12(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        seq = cc.Hold(5.0, value=1.0) + cc.Hold(5.0, value=2.0)
        returned_fig, returned_ax = seq.plot(ax=ax)
        assert returned_ax is ax
        assert returned_fig is fig
        plt.close(fig)


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
