import numpy as np
from scipy import integrate

from climatecritters.core.forcing import Forcing
from climatecritters.signal_models import ENSORechargeOscillator


def _lab09_recharge_deriv(x, t, *pars):
    b = pars[5] * pars[0]
    R = pars[6] * b - pars[2]
    r = pars[3]
    en = pars[1]
    gamma = pars[6]
    alpha = pars[4]
    Af = pars[7]
    Pf = pars[8]
    forcing = Af * np.sin(2.0 * np.pi * t / Pf)
    return [
        R * x[0] + gamma * x[1] - en * (x[1] + b * x[0]) ** 3 + forcing,
        -r * x[1] - alpha * b * x[0],
    ]


def _lab09_recharge_solver(mu=0.7, en=0.0, Af=0.0, Pf=6.0, x0=None, max_time=500 * 6):
    if x0 is None:
        x0 = [0.1, -0.1]
    c = 1.0
    r = 0.25
    alpha = 0.125
    b0 = 2.5
    gamma = 0.75
    pars = mu, en, c, r, alpha, b0, gamma, Af, Pf
    t = np.linspace(0, max_time, int(4 * max_time), endpoint=False)
    x_t = np.asarray(integrate.odeint(_lab09_recharge_deriv, x0, t, pars))
    return t, x_t


class TestSignalModelsENSORechargeOscillator:
    def test_import_and_integrate_t0(self):
        model = ENSORechargeOscillator()
        model.integrate(t_span=(0, 12), y0=[0.1, -0.1], method="RK45")

        assert model.state_variables.dtype.names == ("T", "h")
        assert np.all(np.isfinite(model.state_variables["T"]))
        assert np.all(np.isfinite(model.state_variables["h"]))

    def test_matches_lab09_recharge_solver_t0(self):
        mu = 0.8
        en = 3.0
        Af = 0.03
        Pf = 6.0
        x0 = [0.1, -0.1]
        t_ref, x_ref = _lab09_recharge_solver(mu=mu, en=en, Af=Af, Pf=Pf, x0=x0, max_time=120)

        model = ENSORechargeOscillator(mu=mu, en=en, Af=Af, Pf=Pf)
        model.integrate(
            t_span=(0, 120),
            y0=x0,
            method="LSODA",
            kwargs={"t_eval": t_ref, "rtol": 1e-9, "atol": 1e-11},
        )

        got = np.column_stack([model.state_variables["T"], model.state_variables["h"]])
        assert np.allclose(got, x_ref, rtol=1e-6, atol=1e-6)

    def test_sst_forcing_matches_analytic_at_quarter_period(self):
        """_sst_forcing at t=Pf/4 should equal Af (peak of sine)."""
        Af, Pf = 2.5, 6.0
        model = ENSORechargeOscillator(mu=0.8, en=0.2, Af=Af, Pf=Pf)
        t_peak = Pf / 4.0
        assert np.isclose(model._sst_forcing(t_peak, [0.1, -0.2]), Af)

    def test_enso_like_period_t0(self):
        model = ENSORechargeOscillator(mu=0.8, en=3.0, Af=0.0, Pf=6.0)
        model.integrate(
            t_span=(0, 600),
            y0=[0.1, -0.1],
            method="LSODA",
            kwargs={"t_eval": np.linspace(0, 600, 2400, endpoint=False)},
        )

        T = model.state_variables["T"]
        time_years = model.time * model.tscale
        start = len(T) // 2
        peaks = []
        for i in range(start + 1, len(T) - 1):
            if T[i - 1] < T[i] and T[i] > T[i + 1]:
                peaks.append(i)

        assert len(peaks) >= 3
        period_years = float(np.mean(np.diff(time_years[peaks[-6:]])))
        assert 3.0 <= period_years <= 5.0
