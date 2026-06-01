"""Stocker & Johnsen (2003) thermal bipolar seesaw model.

This module implements a minimum thermodynamic model in which a southern
temperature anomaly responds to a prescribed northern anomaly with a single
timescale:

    dTs/dt = (beta * Tn(t) - Ts) / tau
"""

from __future__ import annotations

import numpy as np

from ..core.pbmodel import PBModel


class Stocker2003BipolarSeesaw(PBModel):
    """Minimum thermodynamic model for the thermal bipolar seesaw.

    A single prognostic southern temperature anomaly ``Ts`` relaxes toward
    a northern temperature signal ``Tn(t)`` scaled by ``beta``:

        dTs/dt = (beta*Tn(t) - Ts) / tau

    Parameters
    ----------
    forcing : pb.core.Forcing or None
        Optional time-varying northern temperature anomaly ``Tn(t)``
        (model units).  If ``None``, the constant ``Tn`` parameter is used.
        Default ``None``.
    var_name : str
        Label for the model output.  Default ``'stocker2003_bipolar_seesaw'``.
    tau : float or callable or pb.core.Forcing
        Thermal equilibration timescale (years).  Must be > 0.  Default 1000.
    beta : float or callable or pb.core.Forcing
        Amplitude ratio relating southern to northern anomaly.  Default -1.0
        (antiphase seesaw).
    Tn : float or callable or pb.core.Forcing
        Constant northern temperature anomaly used when no forcing is
        provided.  Default 0.0.

    Notes
    -----
    The diagnostic variable ``Tn`` (northern temperature) is written by
    ``populate_diagnostics_from_history`` after integration.  State variable
    is ``Ts``.

    References
    ----------
    Stocker, T. F., & Johnsen, S. J. (2003). A minimum thermodynamic model
    for the bipolar seesaw. Paleoceanography, 18(4), 1087.

    Examples
    --------
    ```python
    import numpy as np
    import paleobeasts as pb
    from paleobeasts.signal_models.stocker2003_bipolar_seesaw import (
        Stocker2003BipolarSeesaw,
    )

    import matplotlib.pyplot as plt

    model = Stocker2003BipolarSeesaw(tau=500.0, beta=-1.0)
    model.register_forcing('Tn', pb.core.Forcing(lambda t: 1.0 if (t % 2000) < 1000 else -1.0))
    output = model.integrate(
        t_span=(0, 8000), y0=[0.0], method='RK45'
    )
    ts = output.to_pyleo(var_names=['Ts'])
    ts.plot()
    plt.savefig('docs/reference/figures/Stocker2003BipolarSeesaw_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(
        self,
        var_name="stocker2003_bipolar_seesaw",
        tau=1000.0,
        beta=-1.0,
        Tn=0.0,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["Ts"]
        if diagnostic_variables is None:
            diagnostic_variables = ["Tn"]

        super().__init__(
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.tau = tau
        self.beta = beta
        self.Tn = Tn
        self.param_values = {
            "tau": tau,
            "beta": beta,
            "Tn": Tn,
        }
        self.params = ()

    uses_post_history = True

    def dydt(self, t, x):
        Ts = float(np.asarray(x, dtype=float)[0])
        tau = float(self.get_param_value("tau", t, x))
        if tau <= 0:
            raise ValueError("tau must be > 0.")
        beta = float(self.get_param_value("beta", t, x))
        Tn_t = float(self.get_param_value("Tn", t, x))
        dTsdt = (beta * Tn_t - Ts) / tau
        return [dTsdt]

    def populate_diagnostics_from_history(self, time, history):
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)
        Tn_vals = []
        for t, row in zip(time, history):
            Tn_vals.append(float(self.get_param_value("Tn", t, row)))
        Tn_vals = np.asarray(Tn_vals, dtype=float)
        self.diagnostic_variables = {"Tn": Tn_vals}


class Stocker2003ExtendedSeaIceSeesaw(PBModel):
    """Extended Stocker-style model with reservoir, Southern Ocean, sea-ice, and Antarctic states.

    The model integrates four coupled ODEs with prescribed northern forcing
    ``T_N(t)``:

        tau_R * dT_R/dt     = -(T_R - T_N) + eps_R
        tau_S * dT_S/dt     = kappa*(T_R - T_S) - lambda_S*(T_S - T_S0)
                              + alpha*(1 - A) + eps_S
        tau_A * dA/dt       = -beta*(T_S - T_S0)
                              - gamma*A*(1-A)*(T_S - T_c) + eps_A
        tau_ANT * dT_ANT/dt = delta*(T_S - T_ANT) + eta*(1 - A) + eps_ANT

    Sea-ice area fraction ``A`` is constrained to [0, 1] by suppressing
    the outward derivative at the physical boundaries inside ``dydt``.

    Parameters
    ----------
    forcing : pb.core.Forcing or None
        Optional time-varying northern temperature anomaly ``T_N(t)``.  If
        ``None``, the constant ``T_N`` parameter is used.  Default ``None``.
    var_name : str
        Label for the model output.  Default
        ``'stocker2003_extended_seaice_seesaw'``.
    tau_R : float
        Oceanic reservoir relaxation timescale (years).  Default 300.
    tau_S : float
        Southern Ocean relaxation timescale (years).  Default 1200.
    tau_A : float
        Sea-ice adjustment timescale (years).  Default 100.
    tau_ANT : float
        Antarctic temperature adjustment timescale (years).  Default 20.
    kappa : float
        Advective heat exchange between reservoir and Southern Ocean.
        Default 1.0.
    lambda_S : float
        Linear restoring rate for Southern Ocean temperature.  Default 0.2.
    alpha : float
        Sea-ice insulation effect on Southern Ocean heat flux.  Default 0.3.
    beta : float
        Temperature-driven sea-ice melt rate.  Default 0.2.
    gamma : float
        Nonlinear sea-ice feedback strength.  Default 4.0.
    delta : float
        Southern Ocean to Antarctic heat coupling.  Default 1.0.
    eta : float
        Sea-ice insulation effect on Antarctic temperature.  Default 0.2.
    T_S0 : float
        Reference Southern Ocean temperature.  Default 0.0.
    T_c : float
        Critical temperature for the sea-ice feedback.  Default 0.0.
    T_N : float
        Constant northern temperature when no forcing is provided.
        Default 0.0.
    epsilon_R, epsilon_S, epsilon_A, epsilon_ANT : float
        Constant additive noise / bias terms.  All default to 0.0.

    Notes
    -----
    State variables are ``T_R``, ``T_S``, ``A``, ``T_ANT`` in that order.
    The diagnostic variable ``T_N`` is populated by
    ``populate_diagnostics_from_history``.  All timescales must be > 0.

    Examples
    --------
    ```python
    import paleobeasts as pb
    from paleobeasts.signal_models.stocker2003_bipolar_seesaw import (
        Stocker2003ExtendedSeaIceSeesaw,
    )
    import matplotlib.pyplot as plt


    model = Stocker2003ExtendedSeaIceSeesaw()
    model.register_forcing('T_N', pb.core.Forcing(lambda t: 1.0 if (t % 2000) < 1000 else 0.0))
    output = model.integrate(
        t_span=(0, 10000), y0=[0.0, 0.0, 0.3, 0.0], method='RK45'
    )
    ts = output.to_pyleo(var_names=['T_ANT'])
    ts.plot()
    plt.savefig('docs/reference/figures/Stocker2003ExtendedSeaIceSeesaw_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(
        self,
        var_name="stocker2003_extended_seaice_seesaw",
        tau_R=300.0,
        tau_S=1200.0,
        tau_A=100.0,
        tau_ANT=20.0,
        kappa=1.0,
        lambda_S=0.2,
        alpha=0.3,
        beta=0.2,
        gamma=4.0,
        delta=1.0,
        eta=0.2,
        T_S0=0.0,
        T_c=0.0,
        T_N=0.0,
        epsilon_R=0.0,
        epsilon_S=0.0,
        epsilon_A=0.0,
        epsilon_ANT=0.0,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["T_R", "T_S", "A", "T_ANT"]
        if diagnostic_variables is None:
            diagnostic_variables = ["T_N"]

        super().__init__(
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.tau_R = tau_R
        self.tau_S = tau_S
        self.tau_A = tau_A
        self.tau_ANT = tau_ANT
        self.kappa = kappa
        self.lambda_S = lambda_S
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.T_S0 = T_S0
        self.T_c = T_c
        self.T_N = T_N
        self.epsilon_R = epsilon_R
        self.epsilon_S = epsilon_S
        self.epsilon_A = epsilon_A
        self.epsilon_ANT = epsilon_ANT

        self.param_values = {
            "tau_R": tau_R,
            "tau_S": tau_S,
            "tau_A": tau_A,
            "tau_ANT": tau_ANT,
            "kappa": kappa,
            "lambda_S": lambda_S,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "eta": eta,
            "T_S0": T_S0,
            "T_c": T_c,
            "T_N": T_N,
            "epsilon_R": epsilon_R,
            "epsilon_S": epsilon_S,
            "epsilon_A": epsilon_A,
            "epsilon_ANT": epsilon_ANT,
        }
        self.params = ()

    uses_post_history = True

    def resolve_north(self, t, state):
        return float(self.get_param_value("T_N", t, state))

    def dydt(self, t, x):
        state = np.asarray(x, dtype=float).reshape(-1)
        T_R, T_S, A, T_ANT = [float(v) for v in state]
        A_eff = float(np.clip(A, 0.0, 1.0))

        tau_R = float(self.get_param_value("tau_R", t, state))
        tau_S = float(self.get_param_value("tau_S", t, state))
        tau_A = float(self.get_param_value("tau_A", t, state))
        tau_ANT = float(self.get_param_value("tau_ANT", t, state))
        for name, value in (("tau_R", tau_R), ("tau_S", tau_S), ("tau_A", tau_A), ("tau_ANT", tau_ANT)):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0.")

        kappa = float(self.get_param_value("kappa", t, state))
        lambda_s = float(self.get_param_value("lambda_S", t, state))
        alpha = float(self.get_param_value("alpha", t, state))
        beta = float(self.get_param_value("beta", t, state))
        gamma = float(self.get_param_value("gamma", t, state))
        delta = float(self.get_param_value("delta", t, state))
        eta = float(self.get_param_value("eta", t, state))
        T_S0 = float(self.get_param_value("T_S0", t, state))
        T_c = float(self.get_param_value("T_c", t, state))
        T_N = self.resolve_north(t, state)
        eps_R = float(self.get_param_value("epsilon_R", t, state))
        eps_S = float(self.get_param_value("epsilon_S", t, state))
        eps_A = float(self.get_param_value("epsilon_A", t, state))
        eps_ANT = float(self.get_param_value("epsilon_ANT", t, state))

        dT_R = (-(T_R - T_N) + eps_R) / tau_R
        dT_S = (kappa * (T_R - T_S) - lambda_s * (T_S - T_S0) + alpha * (1.0 - A_eff) + eps_S) / tau_S
        dA = (-beta * (T_S - T_S0) - gamma * A_eff * (1.0 - A_eff) * (T_S - T_c) + eps_A) / tau_A
        dT_ANT = (delta * (T_S - T_ANT) + eta * (1.0 - A_eff) + eps_ANT) / tau_ANT

        if A_eff <= 0.0 and dA < 0.0:
            dA = 0.0
        elif A_eff >= 1.0 and dA > 0.0:
            dA = 0.0

        return [dT_R, dT_S, dA, dT_ANT]

    def populate_diagnostics_from_history(self, time, history):
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)
        Tn_vals = []
        for t, row in zip(time, history):
            Tn_vals.append(self.resolve_north(t, row))
        self.diagnostic_variables = {"T_N": np.asarray(Tn_vals, dtype=float)}

