from __future__ import annotations

import numpy as np

from paleobeasts.core.pbmodel import PBModel


class ENSORechargeOscillator(PBModel):
    """Jin-style ENSO recharge oscillator.

    The implementation follows the lab09 recharge-oscillator worksheet, but
    uses the Paleobeasts ``dydt(self, t, state)`` convention instead of the
    worksheet's ``odeint`` state-first convention.
    """

    def __init__(
        self,
        forcing=None,
        var_name="enso_recharge_oscillator",
        mu=0.7,
        en=0.0,
        Af=0.0,
        Pf=6.0,
        c=1.0,
        r=0.25,
        alpha=0.125,
        b0=2.5,
        gamma=0.75,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["T", "h"]
        if diagnostic_variables is None:
            diagnostic_variables = []

        super().__init__(
            forcing,
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.mu = mu
        self.en = en
        self.Af = Af
        self.Pf = Pf
        self.c = c
        self.r = r
        self.alpha = alpha
        self.b0 = b0
        self.gamma = gamma
        self.tscale = 1.0 / 6.0
        self.param_values = {
            "mu": mu,
            "en": en,
            "Af": Af,
            "Pf": Pf,
            "c": c,
            "r": r,
            "alpha": alpha,
            "b0": b0,
            "gamma": gamma,
        }
        self.params = ()

    def uses_post_history(self):
        return True

    def recharge_components(self, t, state):
        T, h = [float(v) for v in np.asarray(state, dtype=float).reshape(-1)]
        mu = float(self.get_param("mu", t, state))
        en = float(self.get_param("en", t, state))
        c = float(self.get_param("c", t, state))
        r = float(self.get_param("r", t, state))
        alpha = float(self.get_param("alpha", t, state))
        b0 = float(self.get_param("b0", t, state))
        gamma = float(self.get_param("gamma", t, state))
        Af = float(self.get_param("Af", t, state))
        Pf = float(self.get_param("Pf", t, state))

        if Pf == 0.0:
            raise ValueError("Pf must be non-zero.")

        b = b0 * mu
        R = gamma * b - c
        seasonal_forcing = Af * np.sin(2.0 * np.pi * t / Pf)

        dT = R * T + gamma * h - en * (h + b * T) ** 3 + seasonal_forcing
        dh = -r * h - alpha * b * T
        return dT, dh

    def dydt(self, t, state):
        dT, dh = self.recharge_components(t, state)
        return [dT, dh]
