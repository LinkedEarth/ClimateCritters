"""Two-equation bistable Itô SDE for Dansgaard-Oeschger transitions (Melcher et al. 2025)."""

from __future__ import annotations

import numpy as np

from ..core.model import Model

__all__ = ['Melcher25', 'classify_bistable_states']


def _classify_states(db, stadial_threshold, interstadial_threshold):
    """Classify each time step as stadial (1) or interstadial (0) with hysteresis.

    Parameters
    ----------
    db : array-like
        Primary state variable Δb time series.
    stadial_threshold : float
        db drops below this → transition to stadial.
    interstadial_threshold : float
        db rises above this → transition to interstadial.

    Returns
    -------
    states : ndarray
        Binary array: 1 = stadial (cold), 0 = interstadial (warm).
    """
    db = np.asarray(db, dtype=float)
    states = np.zeros(len(db))
    flag = 0
    for i in range(len(db)):
        if db[i] < stadial_threshold:
            flag = 1
        elif db[i] > interstadial_threshold:
            flag = 0
        states[i] = flag
    return states


class Melcher25(Model):
    """Two-equation bistable Itô SDE for stochastic Dansgaard-Oeschger transitions.

    Models the meridional buoyancy gradient Δb and buoyancy flux B as a coupled
    stochastic system that switches between stadial (weak AMOC) and interstadial
    (strong AMOC) states:

        d(db)/dt = -B - |q0 + q1*(db - b0)| * (db - b0)  +  sigma * dW
        dB/dt    = (db + alpha*B - gamma) / tau             +  sigma * dW

    Parameters
    ----------
    var_name : str
        Label for model output. Default is 'bistable'.
    sigma : float
        Noise amplitude applied to both state variables. Default is 0.2, the
        fitted value used in Melcher et al. (2025). The paper gives no formal
        upper bound: [0.15, 0.4] is the range explored in the original
        calibration grid (not a stability limit), and the paper only notes
        qualitatively that "if σ is too high, the model behaves like a random
        walk, leading to abrupt and unrealistic transitions" (Sec. 2.3.5).
    gamma : float or callable or cc.core.Forcing
        External forcing proportional to atmospheric CO2. Bistable behaviour
        occurs near γ ≈ 1.5; constrained to [0.6, 3.0] in Melcher et al.
        (2025, Sec. 2.3.5) to keep the nullcline inside the relevant area of
        the system. If callable, must follow the signature contract in
        ``contracts/signal_model_contract.md``: ``(t)``, ``(t, state)``, or
        ``(t, state, model)`` with the first argument named ``t`` or
        ``time``. Default is 1.5.
    alpha : float or callable or cc.core.Forcing
        Southern Ocean AABW coupling; controls the asymmetry between stadial
        and interstadial durations. Constrained to [-1.0, 0.0] in Melcher
        et al. (2025, Sec. 2.3.5): α > 0 makes the system diverge instead of
        converging, and α < -1 would require unreasonably high noise to
        match observations. If callable, must follow the same signature
        contract as gamma. Default is 0.0.
    b0 : float
        Reference buoyancy gradient that sets the centre of the bistable
        potential well. Default is 0.625 (calibrated from NGRIP,
        Melcher et al. 2025). Changing this value shifts both fixed points
        and may destroy bistability if moved far from the calibrated value.
    q0 : float
        Constant term in the nonlinear restoring force. Default is -9.0
        (calibrated from NGRIP). Together with ``q1`` it controls the depth
        and separation of the two potential wells; large departures from the
        default can eliminate one well and collapse the system to monostable
        behaviour.
    q1 : float
        Linear coefficient in the nonlinear restoring force. Default is 12.0
        (calibrated from NGRIP). Must be positive for the restoring force to
        be bounded; values far from the default alter the transition rates.
    tau : float
        Relaxation time-scale of the buoyancy-flux equation (in model time
        units). Default is 0.902 (calibrated from NGRIP). Very small values
        make B relax almost instantaneously and can suppress oscillations;
        very large values slow the recovery from transitions.

    Notes
    -----
    The parameters ``b0``, ``q0``, ``q1``, and ``tau`` are calibrated from
    NGRIP data (Melcher et al. 2025). Their defaults are the suggested values
    that reproduce realistic Dansgaard-Oeschger statistics. You may change
    them to explore sensitivity or to fit a different dataset, but be aware
    that combinations far from the defaults can suppress bistability entirely.

    All seven parameters (``sigma``, ``gamma``, ``alpha``, ``b0``, ``q0``,
    ``q1``, ``tau``) are registered in ``param_values`` and are therefore
    inspectable via ``model.doc('parameters')`` and ``model.list('parameters')``.
    ``sigma``, ``gamma``, and ``alpha`` additionally support time-varying values
    via callables or ``Forcing`` objects.

    This model sets ``uses_post_history = True``: state variables are derived
    from the full solved trajectory. After every ``integrate()`` call, the
    ``states`` diagnostic variable is auto-populated with the hysteresis
    classification, and ``model.stadial_threshold`` / ``model.interstadial_threshold``
    are set to the Jacobian-derived thresholds. Use ``method='heun_maruyama'``
    for strong order 1.0 accuracy (preferred over ``'euler_maruyama'`` for this
    additive-noise model).

    See also
    --------
    cc.core.Model : Base class.
    classify_bistable_states : Reclassify a Δb signal post-hoc without
        re-running the SDE.

    Examples
    --------
    Generate a synthetic DO record and inspect states

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from climatecritters.model_critters.melcher25 import (
        Melcher25, classify_bistable_states
    )

    model = Melcher25(sigma=0.2, gamma=1.5, alpha=-0.4)
    output = model.integrate(
        t_span=(0, 599.88), y0=[1.0, 0.0],
        method='heun_maruyama', dt=0.012,
        kwargs={'random_seed': 42, 'si': 0.12},
    )

    # Δb time series as a pyleoclim Series — ready to plot or analyse
    ts = output.to_pyleo('db')
    ts.plot()
    plt.savefig('docs/reference/figures/Melcher25_example.png',
            dpi=150, bbox_inches='tight')

    # Stadial/interstadial classification is computed automatically
    states = output.diagnostic_variables['states']   # 1 = cold, 0 = warm
    print('stadial threshold:',      model.stadial_threshold)
    print('interstadial threshold:', model.interstadial_threshold)

    # Count DO events (interstadial onsets)
    n_events = int(np.sum(np.diff(states) == -1))
    print('number of DO events:', n_events)
    ```

    Add a time-varying forcing (e.g. a slow CO2 ramp across a glacial cycle):

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from climatecritters.model_critters.melcher25 import Melcher25

    t_arr = np.linspace(0, 599.88, 5000)
    gamma_ramp = np.linspace(0.8, 3.2, 5000)   # γ increases over time

    model = Melcher25(
        sigma=0.2,
        gamma=lambda t: float(np.interp(t, t_arr, gamma_ramp)),
        alpha=0.0,
    )
    output = model.integrate(
        t_span=(t_arr[0], t_arr[-1]), y0=[1.0, 0.0],
        method='heun_maruyama', dt=0.012,
        kwargs={'random_seed': 0, 'si': 0.12},
    )
    ts = output.to_pyleo('db')
    ts.plot()
    plt.savefig('docs/reference/figures/Melcher25_ramp_example.png',
            dpi=150, bbox_inches='tight')
    ```

    References
    ----------
    Melcher et al. (2025), *Clim. Past*, 21, 115-132. https://cp.copernicus.org/articles/21/115/2025/
    """

    uses_post_history = True

    def __init__(self, var_name='bistable', sigma=0.2, gamma=1.5,
                 alpha=0.0, b0=0.625, q0=-9.0, q1=12.0, tau=0.902,
                 state_variables=None, diagnostic_variables=None,
                 *args, **kwargs):
        if state_variables is None:
            state_variables = ['db', 'B']
        if diagnostic_variables is None:
            diagnostic_variables = ['states']

        super().__init__(var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables, *args, **kwargs)

        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha
        self.b0  = b0
        self.q0  = q0
        self.q1  = q1
        self.tau = tau
        self.param_values = {
            'sigma': sigma,
            'gamma': gamma,
            'alpha': alpha,
            'b0':    b0,
            'q0':    q0,
            'q1':    q1,
            'tau':   tau,
        }
        self.params = ()

        # set after each integrate() by populate_diagnostics_from_history
        self.stadial_threshold     = None
        self.interstadial_threshold = None

    def dydt(self, t, y):
        """Evaluate the deterministic drift of the bistable SDE at time t and state y.

        Called by the solver at each timestep. Because this model sets
        ``uses_post_history = True``, state is not accumulated here as a
        side effect.

        Parameters
        ----------
        t : float
            Current time.
        y : array-like
            Current state vector [db, B].

        Returns
        -------
        dydt : list of float
            Time-derivatives [d(db)/dt, dB/dt].
        """
        gamma = float(self.get_param_value('gamma', t, y))
        alpha = float(self.get_param_value('alpha', t, y))
        db, B = float(y[0]), float(y[1])
        ddb = -B - np.abs(self.q0 + self.q1 * (db - self.b0)) * (db - self.b0)
        dB  = (db + alpha * B - gamma) / self.tau
        return [ddb, dB]

    def sde_noise(self, t, y):
        """Return the diffusion coefficient vector at time t and state y.

        Called by SDE solvers (``euler_maruyama``, ``heun_maruyama``,
        ``milstein``) to scale the Wiener increments. Noise is additive
        (state-independent), which means ``heun_maruyama`` achieves strong
        order 1.0 for this model.

        Parameters
        ----------
        t : float
            Current time.
        y : array-like
            Current state vector [db, B].

        Returns
        -------
        diffusion : ndarray of shape (2,)
            Diffusion coefficient for each state variable.
        """
        sigma = float(self.get_param_value('sigma', t, y))
        return sigma * np.ones(2)

    def populate_diagnostics_from_history(self, time, history):
        """Classify states and compute thresholds from the full solved trajectory.

        Called automatically by ``Model.post_integrate`` after every
        ``integrate()`` call. Populates ``diagnostic_variables['states']``
        with the hysteresis classification and sets
        ``self.stadial_threshold`` / ``self.interstadial_threshold``.

        Parameters
        ----------
        time : array-like
            Time axis of the solution.
        history : ndarray of shape (n_times, 2)
            State trajectory [db, B] at each time step.
        """
        db_vals   = history[:, 0]
        alpha_raw = self.param_values.get('alpha', 0.0)

        alpha_forced = any(name == 'alpha'
                           for specs in self._pre_param_forcings() for name, _ in specs)
        if callable(alpha_raw) or hasattr(alpha_raw, 'get_forcing') or alpha_forced:
            alpha_for_thresh = np.array([
                float(self.call_with_forcings(t, self.get_param_value, 'alpha', t, history[i]))
                for i, t in enumerate(time)
            ])
        else:
            alpha_for_thresh = alpha_raw

        stadial, interstadial = self.compute_stability_thresholds(alpha_for_thresh)
        self.stadial_threshold      = stadial
        self.interstadial_threshold = interstadial

        self.diagnostic_variables = {
            'states': _classify_states(db_vals, stadial, interstadial)
        }

    def compute_stability_thresholds(self, alpha):
        """Compute stadial and interstadial thresholds via Jacobian stability analysis.

        Parameters
        ----------
        alpha : float or array-like
            Southern Ocean AABW coupling parameter. The mean is taken if an
            array is passed (the stadial threshold shifts by less than 0.05
            units across the full alpha range, so the mean is a good
            approximation).

        Returns
        -------
        stadial_threshold : float
            db drops below this → stadial state (1).
        interstadial_threshold : float
            db rises above this → interstadial state (0).

        Notes
        -----
        The lower threshold is the bifurcation point where trace(J) = 0
        (Eq. 7, Melcher et al. 2025):

            b_low = b0 + (-q0 - alpha/tau) / (2*q1)

        The upper threshold is the non-differentiable point of the drift:

            b_nd = b0 - q0/q1

        which is independent of alpha.
        """
        alpha_val = float(np.mean(alpha)) if hasattr(alpha, '__len__') else float(alpha)
        stadial = (self.b0
                   + (-self.q0 - alpha_val / self.tau)
                   / (2.0 * self.q1))
        interstadial = self.b0 - self.q0 / self.q1
        return stadial, interstadial


def classify_bistable_states(signal, alpha, b0=0.625, q0=-9.0, q1=12.0, tau=0.902):
    """Classify a Δb signal into stadial (1) / interstadial (0) states.

    Applies the same hysteresis threshold logic that ``Melcher25``
    uses internally after integration. Useful for reclassifying a Δb signal
    post-hoc — for example after adding proxy noise — without re-running the SDE.

    Parameters
    ----------
    signal : array-like
        Δb signal values to classify.
    alpha : float or array-like
        Southern Ocean AABW coupling value used when the series was generated.
        The mean is taken if an array is passed.
    b0, q0, q1, tau : float
        Model constants. Must match those used during integration if a
        non-default model was run. Defaults match the NGRIP calibration.

    Returns
    -------
    states : ndarray
        Binary array: 1 = stadial (cold), 0 = interstadial (warm).

    See also
    --------
    Melcher25.compute_stability_thresholds : Threshold computation.
    Melcher25.populate_diagnostics_from_history : Auto-classification
        after ``integrate()``.
    """
    model = Melcher25(b0=b0, q0=q0, q1=q1, tau=tau)
    stadial_thresh, interstadial_thresh = model.compute_stability_thresholds(alpha)
    return _classify_states(np.asarray(signal, dtype=float), stadial_thresh, interstadial_thresh)
