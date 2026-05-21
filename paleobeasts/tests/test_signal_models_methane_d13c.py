"""Tests for paleobeasts.signal_models.methane_d13c."""

import numpy as np

from scratchwork.signal import methane_d13c


def _default_y0(total=700.0, delta=-47.0):
    ch4_12, ch4_13 = methane_d13c.MethaneD13C.split_total_and_delta(total, delta)
    return [ch4_12, ch4_12, ch4_13, ch4_13]


class TestSignalModelsMethaneD13CIntegrate:
    def test_integrate_t0(self):
        model = methane_d13c.MethaneD13C()
        model.integrate(t_span=(0, 50), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})

        assert model.state_variables.dtype.names == (
            "ch4_12_nh",
            "ch4_12_sh",
            "ch4_13_nh",
            "ch4_13_sh",
        )
        assert set(model.diagnostic_variables) == {
            "ch4_total_nh",
            "ch4_total_sh",
            "delta13c_nh",
            "delta13c_sh",
        }
        assert np.all(np.isfinite(model.diagnostic_variables["ch4_total_nh"]))
        assert np.all(np.isfinite(model.diagnostic_variables["ch4_total_sh"]))
        assert np.all(np.isfinite(model.diagnostic_variables["delta13c_nh"]))
        assert np.all(np.isfinite(model.diagnostic_variables["delta13c_sh"]))
        assert model.uses_post_history() is True

    def test_ratio_conversion_t0(self):
        delta = -47.0
        ratio = methane_d13c.MethaneD13C.delta_to_ratio(delta)
        delta_back = methane_d13c.MethaneD13C.ratio_to_delta(ratio)
        assert np.isclose(delta, delta_back)

    def test_time_varying_param_matches_constant_t0(self):
        y0 = _default_y0()
        model_const = methane_d13c.MethaneD13C(tau_ex=1.0)
        model_tv = methane_d13c.MethaneD13C(tau_ex=lambda t: 1.0)

        model_const.integrate(t_span=(0, 20), y0=y0, method="euler", kwargs={"dt": 1.0})
        model_tv.integrate(t_span=(0, 20), y0=y0, method="euler", kwargs={"dt": 1.0})

        assert np.allclose(
            model_const.diagnostic_variables["ch4_total_nh"],
            model_tv.diagnostic_variables["ch4_total_nh"],
            rtol=1e-10,
            atol=1e-10,
        )


class TestSignalModelsMethaneD13CSyntheticCases:
    def test_steady_state_restart_t0(self):
        model = methane_d13c.MethaneD13C()
        model.integrate(t_span=(0, 400), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})

        final = [
            model.state_variables["ch4_12_nh"][-1],
            model.state_variables["ch4_12_sh"][-1],
            model.state_variables["ch4_13_nh"][-1],
            model.state_variables["ch4_13_sh"][-1],
        ]
        restart = methane_d13c.MethaneD13C()
        restart.integrate(t_span=(0, 20), y0=final, method="euler", kwargs={"dt": 1.0})

        for name in restart.state_variables.dtype.names:
            assert (
                np.max(np.abs(restart.state_variables[name] - restart.state_variables[name][0]))
                < 1e-2
            )

    def test_symmetric_hemispheres_t0(self):
        equal_shares = {category: 0.5 for category in methane_d13c.SOURCE_CATEGORIES}
        model = methane_d13c.MethaneD13C(source_shares=equal_shares)
        model.integrate(t_span=(0, 100), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})

        assert np.allclose(model.state_variables["ch4_12_nh"], model.state_variables["ch4_12_sh"])
        assert np.allclose(model.state_variables["ch4_13_nh"], model.state_variables["ch4_13_sh"])
        assert np.allclose(model.diagnostic_variables["ch4_total_nh"], model.diagnostic_variables["ch4_total_sh"])
        assert np.allclose(model.diagnostic_variables["delta13c_nh"], model.diagnostic_variables["delta13c_sh"])

    def test_single_source_ordering_t0(self):
        results = {}
        for category in methane_d13c.SOURCE_CATEGORIES:
            strengths = {key: 0.0 for key in methane_d13c.SOURCE_CATEGORIES}
            strengths[category] = 100.0
            model = methane_d13c.MethaneD13C(source_strengths=strengths)
            model.integrate(t_span=(0, 300), y0=_default_y0(total=200.0), method="euler", kwargs={"dt": 1.0})
            results[category] = np.mean(model.diagnostic_variables["delta13c_nh"][-50:])

        assert results["pyrogenic"] > results["geological"]
        assert results["geological"] > results["biogenic"]
        assert results["fossil"] > results["biogenic"]

    def test_removed_analysis_api_t0(self):
        model = methane_d13c.MethaneD13C()

        assert not hasattr(model, "synthetic_base_scenario")
        assert not hasattr(model, "reconstruct_total_sources")
        assert not hasattr(model, "prescribed_hemispheric_sources")
        assert not hasattr(model, "invert_biogenic_pyrogenic")
        assert not hasattr(model, "monte_carlo_inversion")
