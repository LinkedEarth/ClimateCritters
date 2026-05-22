import numpy as np

from ..core.pbmodel import PBModel


class Daisyworld(PBModel):
    """Minimal 0D Daisyworld model with black/white daisy coverage and temperature.

    State variables are:
    - ``Aw``: white daisy fractional coverage
    - ``Ab``: black daisy fractional coverage
    - ``T``: planetary mean temperature (K)

    Parameters can be constants, callables, or ``pb.Forcing`` objects.
    """

    def __init__(self, forcing=None, var_name='daisyworld', alpha_w=0.75, alpha_b=0.25, alpha_g=0.5,
                 gamma=0.3, q=20.0, T_opt=295.0, beta_width=0.003265, S0=1365.0, L=1.0, C=10.0,
                 sigma=5.67051196e-8, state_variables=None, diagnostic_variables=None, *args, **kwargs):
        if state_variables is None:
            state_variables = ['Aw', 'Ab', 'T']
        if diagnostic_variables is None:
            diagnostic_variables = ['A_planet', 'A_bare', 'beta_w', 'beta_b']

        super().__init__(forcing, var_name, state_variables=state_variables,
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
        L_base = self.get_param_value('L', t, x)
        if self.forcing is None:
            return L_base
        return L_base + self.forcing.get_forcing(self.time_util(t))

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
