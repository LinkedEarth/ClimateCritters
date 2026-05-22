import numpy as np

from paleobeasts.core.pbmodel import PBModel


class Roessler(PBModel):
    """Roessler chaotic oscillator.

    Parameters
    ----------
    forcing : optional
        Included for consistency with other signal models. The base dynamics do
        not use forcing directly.

    var_name : str
        Name of the variable being modeled. Default is ``'roessler'``.

    a, b, c : float or callable or object with ``get_forcing``
        Model parameters. Each parameter is resolved through ``get_param`` so
        time-varying callables and ``Forcing`` objects are supported.
    """

    def __init__(
        self,
        forcing,
        var_name='roessler',
        a=0.2,
        b=0.2,
        c=5.7,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ['x', 'y', 'z']
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

        self.a = a
        self.b = b
        self.c = c
        self.param_values = {
            'a': a,
            'b': b,
            'c': c,
        }
        self.params = ()

    def dydt(self, t, state):
        x_val, y_val, z_val = state[0], state[1], state[2]
        a = self.get_param_value('a', t, state)
        b = self.get_param_value('b', t, state)
        c = self.get_param_value('c', t, state)

        dxdt = -y_val - z_val
        dydt = x_val + a * y_val
        dzdt = b + z_val * (x_val - c)

        new_row = np.array([(x_val, y_val, z_val)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        return [dxdt, dydt, dzdt]
