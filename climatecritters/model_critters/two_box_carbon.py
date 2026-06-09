from __future__ import annotations

import numpy as np

from climatecritters.core.ccmodel import CCModel


class TwoBoxCarbon(CCModel):
    """Two-box carbon exchange model with explicit box volumes.

    State variables ``A`` and ``S`` are carbon inventories (mass units) in the
    atmospheric and surface-ocean boxes respectively.  Air-sea exchange is
    driven by the concentration gradient:

        exchange = k * (A/V_atm - S/V_surf)
        dA/dt = -exchange + R - l_s*A
        dS/dt = exchange

    Parameters
    ----------
    var_name : str
        Label for the model output.  Default ``'two_box_carbon'``.
    k : float or callable or cc.core.Forcing
        Air-sea gas exchange rate constant (volume units per time).
        Default 0.2.
    R : float or callable or cc.core.Forcing
        Constant carbon source flux into the atmosphere (mass per time).
        Default 0.0.  Register a time-varying source via
        ``model.register_forcing('R', forcing_obj)``.
    l_s : float or callable or cc.core.Forcing
        First-order atmospheric loss coefficient.  Default 0.0.
    V_atm : float or callable or cc.core.Forcing
        Volume of the atmospheric box (must be > 0).  Default 1.0.
    V_surf : float or callable or cc.core.Forcing
        Volume of the surface-ocean box (must be > 0).  Default 1.0.

    Notes
    -----
    Diagnostic variable ``net_flux`` (the atmospheric tendency ``dA/dt``)
    is computed in ``populate_diagnostics_from_history`` after integration.

    ``V_atm`` and ``V_surf`` must be positive; ``ValueError`` is raised if
    either is ≤ 0.

    Examples
    --------
    ```python
    import climatecritters as cc
    from climatecritters.model_critters.two_box_carbon import TwoBoxCarbon
    import matplotlib.pyplot as plt

    model = TwoBoxCarbon(k=0.1, V_atm=1.0, V_surf=50.0)
    output = model.integrate(
        t_span=(0, 200), y0=[800.0, 38000.0], method='RK45'
    )
    ts = output.to_pyleo(var_names=['A'])
    ts.plot()
    plt.savefig('docs/reference/figures/TwoBoxCarbon_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(
        self,
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
