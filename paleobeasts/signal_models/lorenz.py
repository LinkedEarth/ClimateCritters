import numpy as np

from ..core.pbmodel import PBModel


class Lorenz96(PBModel):
    """Lorenz (1996) single-scale and two-scale atmospheric model.

    A periodic ring of *n* slow-scale variables with quadratic advection and
    constant forcing:

        dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F

    When *J* > 0 a second, faster Y layer of *n* × *J* variables is coupled
    to the slow layer (the two-scale system of Lorenz & Emanuel 1998):

        dX_k/dt = ... - (h*c/b) * sum_j Y_{j,k}
        dY_{j,k}/dt = -c*b * Y_{j+1,k} * (Y_{j+2,k} - Y_{j-1,k}) - c*Y_{j,k} + (h*c/b)*X_k

    Parameters
    ----------
    forcing : pb.core.Forcing or None
        Optional forcing object providing F(t).  If ``None``, the constant
        ``F`` parameter is used as the slow-scale forcing.
    var_name : str
        Label for the model output.  Default ``'lorenz96'``.
    n : int
        Number of slow-scale variables.  Default 40.
    J : int
        Fast variables per slow variable.  ``J=0`` (default) gives the
        single-scale system; ``J>0`` activates the two-scale system.
    F : float or callable or pb.core.Forcing
        Slow-scale forcing amplitude.  Default 8.0.
    h : float
        Coupling coefficient between X and Y layers (two-scale only).
        Default 1.0.
    b : float
        Amplitude ratio Y/X (two-scale only).  Default 10.0.
    c : float
        Timescale ratio Y/X (two-scale only).  Default 10.0.
    exact_rhs : bool
        If ``True``, use global ``np.roll`` on the flattened Y vector,
        matching the original L96_model.py reference implementation.
        Default ``False`` (per-block loop, identical results).

    Notes
    -----
    For the two-scale system (``J>0``) use ``method='rk4'`` with a small
    fixed time step passed directly to ``integrate``::

        output = model.integrate(t_span=..., y0=..., method='rk4',
                                 dt=0.005, kwargs={'si': 0.05})

    Adaptive solvers (RK45) call ``dydt`` at unpredictable sub-steps; the
    fixed-step RK4 avoids this problem.  Plain Euler is too coarse to
    resolve the fast Y dynamics.

    References
    ----------
    Lorenz, E. N. (1996). Predictability: A problem partly solved.
    Lorenz, E. N., & Emanuel, K. A. (1998). J. Atmos. Sci., 55, 399–414.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import paleobeasts as pb
        from paleobeasts.signal_models.lorenz import Lorenz96

        # Single-scale system
        model = Lorenz96(forcing=None, n=40, F=8.0)
        y0 = np.random.randn(40) + 8.0
        output = model.integrate(t_span=(0, 10), y0=y0, method='rk4', dt=0.01)
        ts = output.to_pyleo(var_names=['x0'])

        # Two-scale system
        K, J = 36, 10
        model2 = Lorenz96(forcing=None, n=K, J=J, F=10.0)
        y0_2 = np.concatenate([np.random.randn(K) + 10.0,
                                np.random.randn(K * J) * 0.01])
        output2 = model2.integrate(t_span=(0, 10), y0=y0_2,
                                   method='rk4', dt=0.005,
                                   kwargs={'si': 0.05})
    """

    def __init__(self, forcing=None, var_name='lorenz96', n=40, J=0,
                 F=8.0, h=1.0, b=10.0, c=10.0, exact_rhs=False,
                 state_variables=None, diagnostic_variables=None,
                 *args, **kwargs):
        if state_variables is None:
            x_names = [f'x{k}' for k in range(n)]
            y_names = [f'y{j}' for j in range(n * J)]
            state_variables = x_names + y_names
        if diagnostic_variables is None:
            diagnostic_variables = []

        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables,
                         *args, **kwargs)

        self.n = n
        self.J = J
        self.F = F
        self.h = h
        self.b = b
        self.c = c
        self.exact_rhs = exact_rhs
        self.param_values = {'F': F, 'h': h, 'b': b, 'c': c}
        self.params = ()

    def _forcing_value(self, t, x):
        if self.forcing is None:
            return self.get_param_value('F', t, x)
        return self.forcing.get_forcing(self.time_util(t))

    def dydt(self, t, x):
        x = np.asarray(x, dtype=float)

        if self.J == 0:
            return self._dydt_single(t, x)
        return self._dydt_two_scale(t, x)

    def uses_post_history(self):
        return True

    def _dydt_single(self, t, x):
        x = np.asarray(x, dtype=float)
        F_t = self._forcing_value(t, x)
        n = self.n

        dxdt = np.zeros(n, dtype=float)
        for i in range(n):
            dxdt[i] = (x[(i + 1) % n] - x[i - 2]) * x[i - 1] - x[i] + F_t

        return dxdt.tolist()

    def _dydt_two_scale(self, t, x):
        K, J = self.n, self.J
        X, Y = x[:K], x[K:]

        F_t = self._forcing_value(t, x)
        h = self.get_param_value('h', t, x)
        b = self.get_param_value('b', t, x)
        c = self.get_param_value('c', t, x)

        dX = np.zeros(K, dtype=float)
        dY = np.zeros(K * J, dtype=float)

        Y_reshaped = Y.reshape(K, J)
        coupling = Y_reshaped.sum(axis=1)

        for k in range(K):
            xm1 = X[(k - 1) % K]
            xm2 = X[(k - 2) % K]
            xp1 = X[(k + 1) % K]
            dX[k] = -xm1 * (xm2 - xp1) - X[k] + F_t - (h * c / b) * coupling[k]

        if self.exact_rhs:
            hcb = (h * c) / b
            dY = (-c * b * np.roll(Y, -1) *
                  (np.roll(Y, -2) - np.roll(Y, 1)) -
                  c * Y + hcb * np.repeat(X, J))
        else:
            for k in range(K):
                for j in range(J):
                    jm1 = (j - 1) % J
                    jp1 = (j + 1) % J
                    jp2 = (j + 2) % J
                    yjk = Y_reshaped[k, j]
                    dY[k * J + j] = (-c * b * Y_reshaped[k, jp1] *
                                     (Y_reshaped[k, jp2] - Y_reshaped[k, jm1]) -
                                     c * yjk + (h * c / b) * X[k])

        return np.concatenate([dX, dY]).tolist()


class Lorenz63(PBModel):
    """Lorenz (1963) system.

    A minimal three-variable convection model exhibiting sensitive dependence
    on initial conditions and a strange attractor:

        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Parameters
    ----------
    forcing : pb.core.Forcing or None
        Optional external forcing.  If the forcing value ``f(t)`` is scalar
        it is added to ``dx/dt``; if it is array-like with 3 entries it is
        added to ``(dx/dt, dy/dt, dz/dt)``.
    var_name : str
        Label for the model output.  Default ``'lorenz63'``.
    sigma : float or callable or pb.core.Forcing
        Prandtl number controlling rotation of convective rolls.  Default 10.
    rho : float or callable or pb.core.Forcing
        Rayleigh number (reduced) controlling the buoyancy forcing.
        Default 28.
    beta : float or callable or pb.core.Forcing
        Geometric factor controlling the spatial structure.  Default 8/3.

    Notes
    -----
    The classic strange attractor exists for ``sigma=10``, ``rho=28``,
    ``beta=8/3``.  Time-varying parameters are supported as callables with
    signatures ``(t)``, ``(t, state)``, or ``(t, state, model)``.

    State variables are ``x``, ``y``, ``z`` in that order.

    References
    ----------
    Lorenz, E. N. (1963). J. Atmos. Sci., 20, 130–141.

    Examples
    --------
    .. code-block:: python

        import paleobeasts as pb
        from paleobeasts.signal_models.lorenz import Lorenz63

        model = Lorenz63(forcing=pb.core.Forcing(lambda t: 0.0))
        output = model.integrate(
            t_span=(0, 100), y0=[-8.0, 8.0, 27.0], method='RK45'
        )
        ts = output.to_pyleo(var_names=['x', 'y', 'z'])
    """

    def __init__(self, forcing, var_name='lorenz63', sigma=10.0, rho=28.0, beta=8 / 3,
                 state_variables=None, diagnostic_variables=None, *args, **kwargs):
        if state_variables is None:
            state_variables = ['x', 'y', 'z']
        if diagnostic_variables is None:
            diagnostic_variables = []

        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables, *args, **kwargs)

        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.param_values = {
            'sigma': sigma,
            'rho': rho,
            'beta': beta,
        }
        self.params = ()

    def _forcing_vector(self, t):
        if self.forcing is None:
            return np.zeros(3)

        f_val = self.forcing.get_forcing(self.time_util(t))

        if np.isscalar(f_val):
            return np.array([f_val, 0.0, 0.0])

        f_arr = np.asarray(f_val, dtype=float)
        if f_arr.size != 3:
            raise ValueError("Forcing must be a scalar or an array-like with 3 entries.")

        return f_arr.reshape(3,)

    def dydt(self, t, x):
        x_val, y_val, z_val = x[0], x[1], x[2]
        sigma = self.get_param_value('sigma', t, x)
        rho = self.get_param_value('rho', t, x)
        beta = self.get_param_value('beta', t, x)
        f_vec = self._forcing_vector(t)

        dxdt = sigma * (y_val - x_val) + f_vec[0]
        dydt = x_val * (rho - z_val) - y_val + f_vec[1]
        dzdt = x_val * y_val - beta * z_val + f_vec[2]

        new_row = np.array([(x_val, y_val, z_val)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        return [dxdt, dydt, dzdt]
