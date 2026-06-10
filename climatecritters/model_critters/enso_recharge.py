from __future__ import annotations

import numpy as np

from climatecritters.core.ccmodel import CCModel


def seasonal_forcing(A=0.5, period=6.0):
    """Return a sinusoidal seasonal forcing callable for the ENSO recharge oscillator.

    The returned function computes ``A * sin(2π t / period)`` and can be
    wrapped in a :class:`~climatecritters.core.Forcing` or passed directly to
    :meth:`~climatecritters.core.CCModel.register_forcing`::

        import climatecritters as cc
        from climatecritters.model_critters.enso_recharge import (
            ENSORechargeOscillator, seasonal_forcing)

        model = ENSORechargeOscillator()
        model.register_forcing(
            'T',
            cc.Forcing(seasonal_forcing(A=0.5, period=6.0)),
            attachment_style='additive',
            timing='pre',
        )

    Or equivalently with the factory::

        from climatecritters.utils.forcing import create_sinusoid_forcing

        model.register_forcing(
            'T',
            create_sinusoid_forcing(A=0.5, period=6.0),
            attachment_style='additive',
            timing='pre',
        )

    Parameters
    ----------
    A : float
        Seasonal forcing amplitude.  Default 0.5.
    period : float
        Seasonal forcing period (same units as model time).  Default 6.0.

    Returns
    -------
    callable
        Function ``f(t) -> float``.
    """
    def _func(t):
        return float(A * np.sin(2.0 * np.pi * t / period))
    return _func


class ENSORechargeOscillator(CCModel):
    """Jin-style ENSO recharge oscillator.

    Couples the eastern Pacific SST anomaly ``T`` to the thermocline depth
    anomaly ``h`` via a nonlinear recharge-discharge mechanism:

        dT/dt = R*T + gamma*h - en*(h + b*T)^3
        dh/dt = -r*h - alpha*b*T

    where ``b = b0*mu`` and ``R = gamma*b - c``.

    Seasonal or any other external forcing is added through the standard
    :meth:`~climatecritters.core.CCModel.register_forcing` interface::

        from climatecritters.utils.forcing import create_sinusoid_forcing

        model = ENSORechargeOscillator()
        model.register_forcing(
            'T',
            create_sinusoid_forcing(A=0.5, period=6.0),
            attachment_style='additive',
            timing='pre',
        )

    Parameters
    ----------
    var_name : str
        Label for the model output.  Default ``'enso_recharge_oscillator'``.
    mu : float or callable or cc.core.Forcing
        Bjerknes coupling coefficient.  Default 0.7.
    en : float or callable or cc.core.Forcing
        Nonlinear damping coefficient.  Default 0.0 (linear limit).
    c : float or callable or cc.core.Forcing
        Newtonian cooling rate of SST.  Default 1.0.
    r : float or callable or cc.core.Forcing
        Thermocline recharge damping rate.  Default 0.25.
    alpha : float or callable or cc.core.Forcing
        Wind-stress feedback strength.  Default 0.125.
    b0 : float or callable or cc.core.Forcing
        Background thermocline slope sensitivity.  Default 2.5.
    gamma : float or callable or cc.core.Forcing
        Thermocline feedback onto SST.  Default 0.75.

    Notes
    -----
    State vector: ``[T, h]`` — eastern Pacific SST anomaly and thermocline
    depth anomaly.

    References
    ----------
    Jin, F.-F. (1997). An equatorial ocean recharge paradigm for ENSO.
    J. Atmos. Sci., 54, 811–829.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    import climatecritters as cc
    from climatecritters.model_critters.enso_recharge import ENSORechargeOscillator
    from climatecritters.utils.forcing import create_sinusoid_forcing

    model = ENSORechargeOscillator(mu=0.75)
    model.register_forcing(
        'T',
        create_sinusoid_forcing(A=0.5, period=6.0),
        attachment_style='additive',
        timing='pre',
    )
    output = model.integrate(t_span=(0, 120), y0=[0.5, 0.0], method='RK45')
    ts = output.to_pyleo(var_names=['T'])
    ts.plot()
    plt.savefig('docs/reference/figures/ENSORechargeOscillator_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(
        self,
        var_name="enso_recharge_oscillator",
        mu=0.7,
        en=0.0,
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
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.mu = mu
        self.en = en
        self.c = c
        self.r = r
        self.alpha = alpha
        self.b0 = b0
        self.gamma = gamma
        self.tscale = 1.0 / 6.0
        self.param_values = {
            "mu": mu,
            "en": en,
            "c": c,
            "r": r,
            "alpha": alpha,
            "b0": b0,
            "gamma": gamma,
        }
        self.params = ()

    def uses_post_history(self):
        return True

    def dydt(self, t, state):
        T, h = [float(v) for v in np.asarray(state, dtype=float).reshape(-1)]
        mu = float(self.get_param_value("mu", t, state))
        en = float(self.get_param_value("en", t, state))
        c = float(self.get_param_value("c", t, state))
        r = float(self.get_param_value("r", t, state))
        alpha = float(self.get_param_value("alpha", t, state))
        b0 = float(self.get_param_value("b0", t, state))
        gamma = float(self.get_param_value("gamma", t, state))

        b = b0 * mu
        R = gamma * b - c

        dT = R * T + gamma * h - en * (h + b * T) ** 3
        dh = -r * h - alpha * b * T
        return [dT, dh]
