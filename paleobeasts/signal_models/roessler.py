import numpy as np

from paleobeasts.core.pbmodel import PBModel


class Roessler(PBModel):
    """Roessler chaotic oscillator.

    A three-variable continuous-time system with a single scroll attractor:

        dx/dt = -y - z
        dy/dt = x + a*y
        dz/dt = b + z*(x - c)

    Parameters
    ----------
    forcing : pb.core.Forcing or None
        External forcing applied additively to the equations.  If the value
        at time t is scalar it is added to dx/dt only; if it is a 3-element
        array it is added component-wise.  Default ``None``.
    var_name : str
        Label for the model output.  Default ``'roessler'``.
    a : float or callable or pb.core.Forcing
        Controls the strength of the y-feedback.  Default 0.2.
    b : float or callable or pb.core.Forcing
        Offset in the z equation.  Default 0.2.
    c : float or callable or pb.core.Forcing
        Nonlinear threshold in the z equation.  Default 5.7.

    Notes
    -----
    The canonical chaotic attractor exists near ``a=b=0.2``, ``c=5.7``.
    State variables are ``x``, ``y``, ``z`` in that order.
    Time-varying parameters are resolved through ``get_param_value`` and
    support callables with signatures ``(t)``, ``(t, state)``, or
    ``(t, state, model)``.

    References
    ----------
    Rössler, O. E. (1976). Phys. Lett. A, 57(5), 397–398.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    from paleobeasts.signal_models.roessler import Roessler

    model = Roessler(forcing=None)
    output = model.integrate(
        t_span=(0, 200), y0=[0.1, 0.0, 0.0], method='RK45'
    )
    fig, ax = plt.subplots()
    ax.plot(output.state_variables['x'], output.state_variables['z'],
            lw=0.3, alpha=0.8)
    ax.set_xlabel('x'); ax.set_ylabel('z')
    plt.savefig('docs/reference/figures/Roessler_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(
        self,
        forcing=None,
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

    def _forcing_vector(self, t):
        f_val = self.resolve_forcing(t)
        if np.isscalar(f_val):
            return np.array([f_val, 0.0, 0.0])
        f_arr = np.asarray(f_val, dtype=float)
        if f_arr.size != 3:
            raise ValueError("Forcing must be a scalar or an array-like with 3 entries.")
        return f_arr.reshape(3,)

    def dydt(self, t, state):
        x_val, y_val, z_val = state[0], state[1], state[2]
        a = self.get_param_value('a', t, state)
        b = self.get_param_value('b', t, state)
        c = self.get_param_value('c', t, state)

        f_vec = self._forcing_vector(t)
        dxdt = -y_val - z_val + f_vec[0]
        dydt = x_val + a * y_val + f_vec[1]
        dzdt = b + z_val * (x_val - c) + f_vec[2]

        new_row = np.array([(x_val, y_val, z_val)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        return [dxdt, dydt, dzdt]
