from __future__ import annotations

import numpy as np

from paleobeasts.core.pbmodel import PBModel


class TwoBoxCarbon(PBModel):
    """Two-box carbon exchange model with explicit box volumes.

    State variables ``A`` and ``S`` are carbon inventories in the atmospheric
    and surface-ocean boxes. Air-sea exchange is driven by the difference in
    box concentrations, computed from the explicit box volumes.
    """

    def __init__(
        self,
        forcing=None,
        var_name="two_box_carbon",
        k=0.2,
        R=0.0,
        l_s=0.0,
        V_atm=1.0,
        V_surf=1.0,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["A", "S"]
        if diagnostic_variables is None:
            diagnostic_variables = ["net_flux"]

        super().__init__(
            forcing,
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.k = k
        self.R = R
        self.l_s = l_s
        self.V_atm = V_atm
        self.V_surf = V_surf
        self.param_values = {
            "k": k,
            "R": R,
            "l_s": l_s,
            "V_atm": V_atm,
            "V_surf": V_surf,
        }
        self.params = ()

    def uses_post_history(self):
        return True

    def source_flux(self, t, state):
        if self.forcing is not None:
            return float(self.forcing.get_forcing(self.time_util(t)))
        return float(self.get_param_value("R", t, state))

    def tendencies(self, t, state):
        A_mass, S_mass = [float(v) for v in np.asarray(state, dtype=float).reshape(-1)]
        k = float(self.get_param_value("k", t, state))
        R = self.source_flux(t, state)
        l_s = float(self.get_param_value("l_s", t, state))
        V_atm = float(self.get_param_value("V_atm", t, state))
        V_surf = float(self.get_param_value("V_surf", t, state))

        if V_atm <= 0.0 or V_surf <= 0.0:
            raise ValueError("V_atm and V_surf must be > 0.")

        conc_atm = A_mass / V_atm
        conc_surf = S_mass / V_surf
        exchange_flux = k * (conc_atm - conc_surf)

        dA = -exchange_flux + R - l_s * A_mass
        dS = exchange_flux
        return dA, dS

    def dydt(self, t, state):
        dA, dS = self.tendencies(t, state)
        return [dA, dS]

    def populate_diagnostics_from_history(self, time, history):
        net_flux = np.asarray(
            [self.tendencies(t, row)[0] for t, row in zip(time, history)],
            dtype=float,
        )
        self.diagnostic_variables = {"net_flux": net_flux}
