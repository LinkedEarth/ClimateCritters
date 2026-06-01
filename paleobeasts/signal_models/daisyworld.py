import numpy as np

from ..core.pbmodel import PBModel


class Daisyworld(PBModel):
    """Minimal 0D Daisyworld model with black/white daisy coverage and temperature.

    The model couples daisy population dynamics to a zero-dimensional energy
    balance.  Local temperatures depend on albedo contrasts via:

        T_w = T + q * (A_planet - alpha_w)   # white daisy local temperature
        T_b = T + q * (A_planet - alpha_b)   # black daisy local temperature

    and the prognostic equations are:

        dAw/dt = Aw * (A_bare * beta_w - gamma)
        dAb/dt = Ab * (A_bare * beta_b - gamma)
        C * dT/dt = S0*L/4 * (1 - A_planet) - sigma * T^4

    Parameters
    ----------
    forcing : pb.core.Forcing or None
        Optional luminosity perturbation added to ``L``.  Default ``None``.
    var_name : str
        Label for the model output.  Default ``'daisyworld'``.
    alpha_w : float or callable or pb.core.Forcing
        White daisy albedo.  Default 0.75.
    alpha_b : float or callable or pb.core.Forcing
        Black daisy albedo.  Default 0.25.
    alpha_g : float or callable or pb.core.Forcing
        Bare-ground albedo.  Default 0.5.
    gamma : float or callable or pb.core.Forcing
        Daisy death rate (fraction per unit time).  Default 0.3.
    q : float or callable or pb.core.Forcing
        Local temperature sensitivity to albedo contrast (K).  Default 20.0.
    T_opt : float or callable or pb.core.Forcing
        Optimal daisy growth temperature (K).  Default 295.0.
    beta_width : float or callable or pb.core.Forcing
        Parabolic growth-rate width parameter.  Default 0.003265.
    S0 : float or callable or pb.core.Forcing
        Solar constant (W m\ :sup:`-2`).  Default 1365.0.
    L : float or callable or pb.core.Forcing
        Normalized stellar luminosity (1.0 = present Sun).  Default 1.0.
    C : float or callable or pb.core.Forcing
        Planetary heat capacity (effective, in model units).  Default 10.0.
    sigma : float or callable or pb.core.Forcing
        Stefan-Boltzmann constant (W m\ :sup:`-2` K\ :sup:`-4`).
        Default 5.67051196e-8.

    Notes
    -----
    State variables are ``Aw``, ``Ab``, ``T`` in that order.
    Diagnostic variables populated during integration are ``A_planet``,
    ``A_bare``, ``beta_w``, and ``beta_b``.

    Daisy area fractions are soft-clipped to [0, 1] inside ``dydt`` to
    avoid unphysical growth; the solver may transiently produce small
    negative values which are set to zero for tendency calculations.

    References
    ----------
    Watson, A. J., & Lovelock, J. E. (1983). Biological homeostasis of the
    global environment: The parable of Daisyworld. Tellus B, 35(4), 284–289.

    Examples
    --------

    ```python
    import paleobeasts as pb
    from paleobeasts.signal_models.daisyworld import Daisyworld
    import matplotlib.pyplot as plt

    model = Daisyworld(L=0.9)
    output = model.integrate(
        t_span=(0, 500), y0=[0.2, 0.2, 295.0], method='RK45'
    )
    ts = output.to_pyleo(var_names=['T'])
    ts.plot()
    plt.savefig('docs/reference/figures/Daisyworld_example.png',
                dpi=150, bbox_inches='tight')
    ```
    
    """

    def __init__(self, var_name='daisyworld', alpha_w=0.75, alpha_b=0.25, alpha_g=0.5,
                 gamma=0.3, q=20.0, T_opt=295.0, beta_width=0.003265, S0=1365.0, L=1.0, C=10.0,
                 sigma=5.67051196e-8, state_variables=None, diagnostic_variables=None, *args, **kwargs):
        if state_variables is None:
            state_variables = ['Aw', 'Ab', 'T']
        if diagnostic_variables is None:
            diagnostic_variables = ['A_planet', 'A_bare', 'beta_w', 'beta_b']

        super().__init__(var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables, *args, **kwargs)

        self.alpha_w = alpha_w
        self.alpha_b = alpha_b
        self.alpha_g = alpha_g
        self.gamma = gamma
        self.q = q
        self.T_opt = T_opt
        self.beta_width = beta_width
        self.S0 = S0
        self.L = L
        self.C = C
        self.sigma = sigma
        self.param_values = {
            'alpha_w': alpha_w,
            'alpha_b': alpha_b,
            'alpha_g': alpha_g,
            'gamma': gamma,
            'q': q,
            'T_opt': T_opt,
            'beta_width': beta_width,
            'S0': S0,
            'L': L,
            'C': C,
            'sigma': sigma,
        }
        self.params = ()

    def _luminosity(self, t, x):
        return self.get_param_value('L', t, x)

    def _growth(self, T_local, t, x):
        T_opt = self.get_param_value('T_opt', t, x)
        beta_width = self.get_param_value('beta_width', t, x)
        growth = 1.0 - beta_width * (T_opt - T_local) ** 2
        return np.maximum(0.0, growth)

    def dydt(self, t, x):
        Aw = float(x[0])
        Ab = float(x[1])
        T = float(x[2])

        # Keep physically meaningful area fractions in tendency calculations.
        Aw_eff = max(Aw, 0.0)
        Ab_eff = max(Ab, 0.0)
        total = Aw_eff + Ab_eff
        if total > 1.0 and total > 0.0:
            Aw_eff = Aw_eff / total
            Ab_eff = Ab_eff / total
            total = 1.0
        A_bare = 1.0 - total

        alpha_w = self.get_param_value('alpha_w', t, x)
        alpha_b = self.get_param_value('alpha_b', t, x)
        alpha_g = self.get_param_value('alpha_g', t, x)
        gamma = self.get_param_value('gamma', t, x)
        q = self.get_param_value('q', t, x)
        S0 = self.get_param_value('S0', t, x)
        C = self.get_param_value('C', t, x)
        sigma = self.get_param_value('sigma', t, x)
        L_eff = self._luminosity(t, x)

        A_planet = Aw_eff * alpha_w + Ab_eff * alpha_b + A_bare * alpha_g
        T_w = T + q * (A_planet - alpha_w)
        T_b = T + q * (A_planet - alpha_b)
        beta_w = self._growth(T_w, t, x)
        beta_b = self._growth(T_b, t, x)

        dAwdt = Aw_eff * (A_bare * beta_w - gamma)
        dAbdt = Ab_eff * (A_bare * beta_b - gamma)

        absorbed = S0 * L_eff * (1.0 - A_planet) / 4.0
        emitted = sigma * (T ** 4)
        dTdt = (absorbed - emitted) / C

        new_row = np.array([(Aw, Ab, T)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        self.diagnostic_variables['A_planet'].append(A_planet)
        self.diagnostic_variables['A_bare'].append(A_bare)
        self.diagnostic_variables['beta_w'].append(beta_w)
        self.diagnostic_variables['beta_b'].append(beta_b)

        return [dAwdt, dAbdt, dTdt]
