"""Tests for paleobeasts.signal_models.stocker2003_bipolar_seesaw."""

import numpy as np
import paleobeasts as pb

from paleobeasts.signal_models import Stocker2003BipolarSeesaw


class TestSignalModelsStocker2003Integrate:
    def test_integrate_forced_t0(self):
        time = np.linspace(0, 4000, 401)
        north = np.sin(2.0 * np.pi * time / 1200.0)
        forcing = pb.Forcing(data=north, time=time, interpolation="linear")

        model = Stocker2003BipolarSeesaw(tau=1000.0, beta=-1.0)
        model.register_forcing('Tn', forcing)
        model.integrate(t_span=(0, 4000), y0=[0.0], method="euler", dt=10.0)

        assert model.state_variables.dtype.names == ("Ts",)
        assert "Tn" in model.diagnostic_variables
        assert np.all(np.isfinite(model.state_variables["Ts"]))
        assert np.all(np.isfinite(model.diagnostic_variables["Tn"]))

    def test_sign_sanity_t0(self):
        model = Stocker2003BipolarSeesaw(tau=1000.0, beta=-1.0, Tn=1.0)
        dTs = model.dydt(0.0, np.array([0.0]))[0]
        assert dTs < 0.0

    def test_Tn_param_fallback_t0(self):
        model = Stocker2003BipolarSeesaw(Tn=1.5)
        dTs = model.dydt(0.0, np.array([0.0]))[0]
        assert np.isclose(dTs, (-1.0 * 1.5 - 0.0) / 1000.0)
