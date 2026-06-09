from __future__ import annotations

import numpy as np

from climatecritters.core.ccmodel import CCModel


def seasonal_forcing(t, Af=0.5, Pf=6.0):
    """Standard sinusoidal seasonal forcing for the ENSO recharge oscillator.

    Returns the seasonal SST forcing term ``Af * sin(2π t / Pf)``, which can
    be wrapped in a ``Forcing`` object and registered on the model::

        import climatecritters as cc
        from climatecritters.signal_models.enso_recharge import (
            ENSORechargeOscillator, seasonal_forcing)
        from functools import partial

        model = ENSORechargeOscillator()
        sf = partial(seasonal_forcing, Af=0.5, Pf=6.0)
        model.register_forcing('T', cc.core.Forcing(sf),
                               attachment_style='additive', timing='pre')

    Parameters
    ----------
    t : float
        Time (model units, e.g. months).
    Af : float
        Seasonal forcing amplitude.  Default 0.5.
    Pf : float
        Seasonal forcing period (same units as ``t``).  Default 6.0.

    Returns
    -------
    float
        Seasonal forcing value at time ``t``.
    """
    return Af * np.sin(2.0 * np.pi * t / Pf)


class ENSORechargeOscillator(CCModel):
    """Jin-style ENSO recharge oscillator.

    Couples the eastern Pacific SST anomaly ``T`` to the thermocline depth
    anomaly ``h`` via a nonlinear recharge-discharge mechanism:

        dT/dt = R*T + gamma*h - en*(h + b*T)^3 + Af*sin(2*pi*t/Pf)
        dh/dt = -r*h - alpha*b*T

    where ``b = b0*mu`` and ``R = gamma*b - c``.

    By default ``Af=0`` so the model runs without any seasonal forcing.
    To add the standard seasonal cycle, register it externally::

        from functools import partial
        sf = partial(seasonal_forcing, Af=0.5, Pf=6.0)
        model.register_forcing('T', cc.core.Forcing(sf),
                               attachment_style='additive', timing='pre')

    Alternatively, set ``Af`` to a non-zero value to drive seasonal forcing
    internally through the ``_sst_forcing`` helper without registering an
    external forcing.

    Parameters
    ----------
    var_name : str
        Label for the model output.  Default ``'enso_recharge_oscillator'``.
    mu : float or callable or cc.core.Forcing
        Bjerknes coupling coefficient.  Default 0.7.
    en : float or callable or cc.core.Forcing
        Nonlinear damping coefficient.  Default 0.0 (linear limit).
    Af : float or callable or cc.core.Forcing
        Internal seasonal forcing amplitude.  Default 0.0 (no seasonal forcing).
    Pf : float or callable or cc.core.Forcing
        Internal seasonal forcing period (model time units).  Default 6.0.
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
    The internal time scale is ``tscale = 1/6`` (months to years mapping);
    this is baked in but does not affect ``t`` directly as the equations are
    already in the dimensionless form used in the worksheet.

    State vector: ``[T, h]`` — eastern Pacific SST anomaly and thermocline
    depth anomaly.  ``Pf`` must be non-zero when ``Af != 0`` (raises
    ``ValueError`` otherwise).

    References
    ----------
    Jin, F.-F. (1997). An equatorial ocean recharge paradigm for ENSO.
    J. Atmos. Sci., 54, 811–829.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    from climatecritters.signal_models.enso_recharge import ENSORechargeOscillator

    model = ENSORechargeOscillator(mu=0.75, Af=0.5, Pf=6.0)
    output = model.integrate(
        t_span=(0, 120), y0=[0.5, 0.0], method='RK45'
    )
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

    def _sst_forcing(self, t, state):
        Af = float(self.get_param_value("Af", t, state))
        if Af == 0.0:
            return 0.0
        Pf = float(self.get_param_value("Pf", t, state))
        if Pf == 0.0:
            raise ValueError("Pf must be non-zero when Af != 0.")
        return float(Af * np.sin(2.0 * np.pi * t / Pf))

    def recharge_components(self, t, state):
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
        forcing_term = self._sst_forcing(t, state)

        dT = R * T + gamma * h - en * (h + b * T) ** 3 + forcing_term
        dh = -r * h - alpha * b * T
        return dT, dh

    def dydt(self, t, state):
        dT, dh = self.recharge_components(t, state)
        return [dT, dh]
