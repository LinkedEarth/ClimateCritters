from __future__ import annotations

import numpy as np

from ..core.pbmodel import PBModel


class Stommel(PBModel):
    """Minimal two-box Stommel thermohaline circulation model.

    State variables are temperature contrast ``T`` and salinity contrast ``S``.
    The overturning strength is parameterized as:

    ``q = k * (alpha * T - beta * S)``

    and the prognostic system is:

    ``dT/dt = -lambda_T * (T - T_star) - |q| * T + f_T(t)``
    ``dS/dt = E - lambda_S * (S - S_star) - |q| * S + f_S(t)``

    Parameters
    ----------
    forcing : pb.Forcing or None
        Optional external forcing. If scalar, it is added to salinity tendency
        (freshwater forcing term). If array-like with 2 entries, it is added to
        ``(dT/dt, dS/dt)``.

    var_name : str
        Name of the modeled quantity. Default is ``'stommel'``.

    alpha, beta, k, E, lambda_T, lambda_S, T_star, S_star : float or callable or pb.Forcing
        Model parameters. Can be constants, callables, or Forcing objects.
    """

    def __init__(self, forcing=None, var_name='stommel', alpha=1.0, beta=1.0, k=1.0, E=0.0,
                 lambda_T=1.0, lambda_S=1.0, T_star=1.0, S_star=0.0, state_variables=None,
                 diagnostic_variables=None, *args, **kwargs):
        if state_variables is None:
            state_variables = ['T', 'S']
        if diagnostic_variables is None:
            diagnostic_variables = ['q']

        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables, *args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.E = E
        self.lambda_T = lambda_T
        self.lambda_S = lambda_S
        self.T_star = T_star
        self.S_star = S_star
        self.param_values = {
            'alpha': alpha,
            'beta': beta,
            'k': k,
            'E': E,
            'lambda_T': lambda_T,
            'lambda_S': lambda_S,
            'T_star': T_star,
            'S_star': S_star,
        }
        self.params = ()

    def _forcing_vector(self, t):
        if self.forcing is None:
            return np.zeros(2)

        f_val = self.forcing.get_forcing(self.time_util(t))
        if np.isscalar(f_val):
            return np.array([0.0, float(f_val)])

        f_arr = np.asarray(f_val, dtype=float)
        if f_arr.size != 2:
            raise ValueError("Forcing must be a scalar or an array-like with 2 entries.")
        return f_arr.reshape(2,)

    def overturning(self, t, x):
        T, S = x[0], x[1]
        alpha = self.get_param_value('alpha', t, x)
        beta = self.get_param_value('beta', t, x)
        k = self.get_param_value('k', t, x)
        return k * (alpha * T - beta * S)

    def uses_post_history(self):
        return True

    def dydt(self, t, x):
        T, S = x[0], x[1]
        q = self.overturning(t, x)
        adv = np.abs(q)

        E = self.get_param_value('E', t, x)
        lambda_T = self.get_param_value('lambda_T', t, x)
        lambda_S = self.get_param_value('lambda_S', t, x)
        T_star = self.get_param_value('T_star', t, x)
        S_star = self.get_param_value('S_star', t, x)
        f_vec = self._forcing_vector(t)

        dTdt = -lambda_T * (T - T_star) - adv * T + f_vec[0]
        dSdt = E - lambda_S * (S - S_star) - adv * S + f_vec[1]

        return [dTdt, dSdt]

    def populate_diagnostics_from_history(self, time, history):
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)
        q_vals = np.empty(len(time))
        for i, (t, row) in enumerate(zip(time, history)):
            q_vals[i] = self.overturning(t, row)
        self.diagnostic_variables = {'q': q_vals}
