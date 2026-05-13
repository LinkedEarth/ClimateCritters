import numpy as np

from ..core.pbmodel import PBModel
from ..utils.solver import Solution


class Lorenz96(PBModel):
    """Lorenz (1996) system.

    Parameters
    ----------
    forcing : pb.Forcing or None
        Forcing object providing F(t). If None, the constant F parameter is used.

    var_name : str
        Name of the variable being modeled. Default is 'lorenz96'.

    n : int
        Number of state variables. Default is 40.

    F : float or callable or pb.Forcing
        Constant forcing when forcing is None, or a time-varying parameter. Default is 8.

    Notes
    -----
    If ``forcing`` is provided, it takes precedence over ``F``. Otherwise ``F`` can be
    a constant, callable, or ``pb.Forcing`` for time-varying behavior.
    """

    def __init__(self, forcing=None, var_name='lorenz96', n=40, F=8.0,
                 state_variables=None, diagnostic_variables=None, *args, **kwargs):
        if state_variables is None:
            state_variables = [f'x{i}' for i in range(n)]
        if diagnostic_variables is None:
            diagnostic_variables = []

        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables, *args, **kwargs)

        self.n = n
        self.F = F
        self.param_values = {
            'F': F,
        }
        self.params = ()

    def _forcing_value(self, t, x):
        if self.forcing is None:
            return self.get_param('F', t, x)
        return self.forcing.get_forcing(self.time_util(t))

    def dydt(self, t, x):
        x = np.asarray(x, dtype=float)
        F_t = self._forcing_value(t, x)
        n = self.n

        dxdt = np.zeros(n, dtype=float)
        for i in range(n):
            dxdt[i] = (x[(i + 1) % n] - x[i - 2]) * x[i - 1] - x[i] + F_t

        new_row = np.array([tuple(x)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        return dxdt.tolist()


# =============================================================================
# OLD Lorenz96TwoScale — commented out 2026-05-13
# Replaced by the new version below that inherits from Lorenz96.
# Kept for reference and troubleshooting; _state_to_arrays and run were
# identified as redundant (run bypasses the PBModel post-integration API;
# _state_to_arrays duplicates structured array field access).
# =============================================================================

# class Lorenz96TwoScale(PBModel):
#     """Lorenz (1996) two-time scale system.
#
#     Parameters
#     ----------
#     forcing : pb.Forcing or None
#         Optional forcing object providing F(t). If None, the constant F parameter is used.
#
#     var_name : str
#         Name of the variable being modeled. Default is 'lorenz96_two_scale'.
#
#     K : int
#         Number of global-scale variables X. Default is 36.
#
#     J : int
#         Number of local-scale variables Y per global-scale variable. Default is 10.
#
#     F : float or callable or pb.Forcing
#         Constant forcing when forcing is None, or a time-varying parameter. Default is 10.
#
#     h : float or callable or pb.Forcing
#         Coupling coefficient between X and Y. Default is 1.
#
#     b : float or callable or pb.Forcing
#         Ratio of amplitudes between X and Y. Default is 10.
#
#     c : float or callable or pb.Forcing
#         Time-scale ratio between X and Y. Default is 10.
#
#     Notes
#     -----
#     To reproduce the original fixed-step RK4 integration used in the reference
#     notebook, call:
#         ``integrate(method='l96_rk4', kwargs={'dt': dt, 'si': si})``
#     where ``dt`` is the internal integration step and ``si`` is the sampling
#     interval.
#
#     Set ``exact_rhs=True`` to match the original `L96_model.py` tendency for the
#     fast variables, which uses global rolling on the flattened Y state.
#     """
#
#     def __init__(self, forcing=None, var_name='lorenz96_two_scale', K=36, J=10,
#                  F=10.0, h=1.0, b=10.0, c=10.0, exact_rhs=False,
#                  state_variables=None, diagnostic_variables=None, *args, **kwargs):
#         if state_variables is None:
#             x_names = [f'x{k}' for k in range(K)]
#             y_names = [f'y{j}' for j in range(K * J)]
#             state_variables = x_names + y_names
#         if diagnostic_variables is None:
#             diagnostic_variables = []
#
#         super().__init__(forcing, var_name, state_variables=state_variables,
#                          diagnostic_variables=diagnostic_variables, *args, **kwargs)
#
#         self.K = K
#         self.J = J
#         self.F = F
#         self.h = h
#         self.b = b
#         self.c = c
#         self.exact_rhs = exact_rhs
#         self.param_values = {
#             'F': F,
#             'h': h,
#             'b': b,
#             'c': c,
#         }
#         self.params = ()
#
#     def _forcing_value(self, t, x):
#         if self.forcing is None:
#             return self.get_param('F', t, x)
#         return self.forcing.get_forcing(self.time_util(t))
#
#     def _split_state(self, x):
#         x = np.asarray(x, dtype=float)
#         X = x[:self.K]
#         Y = x[self.K:]
#         return X, Y
#
#     def _state_to_arrays(self):
#         state = self.state_variables
#         X = np.column_stack([state[f'x{i}'] for i in range(self.K)])
#         Y = np.column_stack([state[f'y{i}'] for i in range(self.K * self.J)])
#         t = self.time
#         return X, Y, t
#
#     def dydt(self, t, x):
#         X, Y = self._split_state(x)
#
#         F_t = self._forcing_value(t, x)
#         h = self.get_param('h', t, x)
#         b = self.get_param('b', t, x)
#         c = self.get_param('c', t, x)
#
#         K = self.K
#         J = self.J
#
#         dX = np.zeros(K, dtype=float)
#         dY = np.zeros(K * J, dtype=float)
#
#         Y_reshaped = Y.reshape(K, J)
#         coupling = Y_reshaped.sum(axis=1)
#
#         for k in range(K):
#             xm1 = X[(k - 1) % K]
#             xm2 = X[(k - 2) % K]
#             xp1 = X[(k + 1) % K]
#             dX[k] = -xm1 * (xm2 - xp1) - X[k] + F_t - (h * c / b) * coupling[k]
#
#         if self.exact_rhs:
#             hcb = (h * c) / b
#             dY = (-c * b * np.roll(Y, -1) *
#                   (np.roll(Y, -2) - np.roll(Y, 1)) -
#                   c * Y + hcb * np.repeat(X, J))
#         else:
#             for k in range(K):
#                 for j in range(J):
#                     jm1 = (j - 1) % J
#                     jp1 = (j + 1) % J
#                     jp2 = (j + 2) % J
#                     yjk = Y_reshaped[k, j]
#                     dY[k * J + j] = (-c * b * Y_reshaped[k, jp1] *
#                                      (Y_reshaped[k, jp2] - Y_reshaped[k, jm1]) -
#                                      c * yjk + (h * c / b) * X[k])
#
#         new_row = np.array([tuple(np.concatenate([X, Y]))], dtype=self.dtypes)
#         self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
#         if t > 0:
#             self.time.append(t)
#
#         return np.concatenate([dX, dY]).tolist()
#
#     def integrate(self, t_span=None, y0=None, method='RK45', kwargs=None, run_name=None):
#         """Integrate the two-scale system.
#
#         Use method='l96_rk4' to reproduce the fixed-step RK4 integration
#         used in the original Lorenz-96 two-scale reference implementation.
#         """
#
#         if method != 'l96_rk4':
#             return super().integrate(t_span=t_span, y0=y0, method=method, kwargs=kwargs, run_name=run_name)
#
#         if t_span is None or y0 is None:
#             raise ValueError("t_span and y0 must be provided for l96_rk4 integration.")
#
#         kwargs = kwargs or {}
#         if 'dt' not in kwargs or 'si' not in kwargs:
#             raise ValueError("kwargs must include 'dt' and 'si' for l96_rk4 integration.")
#
#         dt = float(kwargs['dt'])
#         si = float(kwargs['si'])
#         t0, t1 = float(t_span[0]), float(t_span[1])
#         total_time = t1 - t0
#
#         if total_time <= 0:
#             raise ValueError("t_span must have t_span[1] > t_span[0].")
#
#         nt = int(round(total_time / si))
#         if abs(nt * si - total_time) > 1e-12:
#             raise ValueError("t_span length must be an integer multiple of si.")
#
#         if si < dt:
#             dt = si
#             ns = 1
#         else:
#             ns = int(round(si / dt))
#             if abs(ns * dt - si) > 1e-12:
#                 raise ValueError("si must be an integer multiple of dt.")
#
#         self.t_span = t_span
#         self.y0 = y0
#         self.method = method
#         self.kwargs = kwargs
#
#         if len(self.state_variables_names) > 0:
#             dtype = [(var, float) for var in self.state_variables_names]
#             self.dtypes = dtype
#
#         y0_arr = np.asarray(y0, dtype=float)
#         history = np.zeros((nt + 1, y0_arr.size), dtype=float)
#         time = np.zeros(nt + 1, dtype=float)
#
#         history[0] = y0_arr
#         time[0] = t0
#
#         X, Y = self._split_state(y0_arr)
#
#         def rhs(t, X_in, Y_in):
#             F_t = self._forcing_value(t, np.concatenate([X_in, Y_in]))
#             h = self.get_param('h', t, X_in)
#             b = self.get_param('b', t, X_in)
#             c = self.get_param('c', t, X_in)
#
#             K = self.K
#             J = self.J
#
#             dX = np.zeros(K, dtype=float)
#             dY = np.zeros(K * J, dtype=float)
#
#             Y_reshaped = Y_in.reshape(K, J)
#             coupling = Y_reshaped.sum(axis=1)
#
#             for k in range(K):
#                 xm1 = X_in[(k - 1) % K]
#                 xm2 = X_in[(k - 2) % K]
#                 xp1 = X_in[(k + 1) % K]
#                 dX[k] = -xm1 * (xm2 - xp1) - X_in[k] + F_t - (h * c / b) * coupling[k]
#
#             if self.exact_rhs:
#                 hcb = (h * c) / b
#                 dY = (-c * b * np.roll(Y_in, -1) *
#                       (np.roll(Y_in, -2) - np.roll(Y_in, 1)) -
#                       c * Y_in + hcb * np.repeat(X_in, J))
#             else:
#                 for k in range(K):
#                     for j in range(J):
#                         jm1 = (j - 1) % J
#                         jp1 = (j + 1) % J
#                         jp2 = (j + 2) % J
#                         yjk = Y_reshaped[k, j]
#                         dY[k * J + j] = (-c * b * Y_reshaped[k, jp1] *
#                                          (Y_reshaped[k, jp2] - Y_reshaped[k, jm1]) -
#                                          c * yjk + (h * c / b) * X_in[k])
#
#             return dX, dY
#
#         for n in range(nt):
#             base_t = t0 + n * si
#             for s in range(ns):
#                 t_curr = base_t + s * dt
#                 dX1, dY1 = rhs(t_curr, X, Y)
#                 dX2, dY2 = rhs(t_curr + 0.5 * dt, X + 0.5 * dt * dX1, Y + 0.5 * dt * dY1)
#                 dX3, dY3 = rhs(t_curr + 0.5 * dt, X + 0.5 * dt * dX2, Y + 0.5 * dt * dY2)
#                 dX4, dY4 = rhs(t_curr + dt, X + dt * dX3, Y + dt * dY3)
#
#                 X = X + (dt / 6.0) * ((dX1 + dX4) + 2.0 * (dX2 + dX3))
#                 Y = Y + (dt / 6.0) * ((dY1 + dY4) + 2.0 * (dY2 + dY3))
#
#             history[n + 1] = np.concatenate([X, Y])
#             time[n + 1] = t0 + (n + 1) * si
#
#         state = np.zeros(nt + 1, dtype=self.dtypes)
#         for i, var in enumerate(self.state_variables_names):
#             state[var] = history[:, i]
#
#         self.state_variables = state
#         self.time = time
#         self.solution = Solution(time, history)
#         self.run_name = run_name if run_name is not None else f'{self.method}, dt={dt}, si={si}'
#
#     def run(self, si, total_time, y0, dt=None, method='euler'):
#         """Integrate and return X, Y, t arrays sampled at interval si."""
#         if dt is None:
#             dt = si
#
#         if method == 'l96_rk4':
#             self.integrate(t_span=(0, total_time), y0=y0, method='l96_rk4', kwargs={'dt': dt, 'si': si})
#             X, Y, t = self._state_to_arrays()
#             return X, Y, t
#
#         if method == 'euler':
#             self.integrate(t_span=(0, total_time), y0=y0, method=method, kwargs={'dt': dt})
#             X, Y, t = self._state_to_arrays()
#             if si is not None and dt is not None:
#                 step = int(round(si / dt))
#                 if step > 1:
#                     X = X[::step]
#                     Y = Y[::step]
#                     t = t[::step]
#         else:
#             self.integrate(t_span=(0, total_time), y0=y0, method=method)
#             if si is not None:
#                 t_eval = np.arange(0, total_time + si, si)
#                 self.reframe_time_axis(t_eval, update_state=True)
#             X, Y, t = self._state_to_arrays()
#
#         return X, Y, t


class Lorenz96TwoScale(Lorenz96):
    """Lorenz (1996) two-time scale system.

    Extends ``Lorenz96`` with J fast variables Y per slow variable X. The slow
    (X) equation is identical to the single-scale system with an added coupling
    term; setting J=0 recovers the single-scale model exactly.

    Parameters
    ----------
    forcing : pb.Forcing or None
        Optional forcing object providing F(t). If None, the constant F parameter is used.

    var_name : str
        Default is 'lorenz96_two_scale'.

    K : int
        Number of slow-scale variables X. Default is 36.

    J : int
        Number of fast-scale variables Y per X variable. Default is 10.

    F : float or callable or pb.Forcing
        Slow-scale forcing. Default is 10.

    h : float or callable or pb.Forcing
        Coupling coefficient between X and Y. Default is 1.

    b : float or callable or pb.Forcing
        Amplitude ratio Y/X. Default is 10.

    c : float or callable or pb.Forcing
        Timescale ratio Y/X. Default is 10.

    exact_rhs : bool
        If True, use global np.roll for the Y tendency (matches the original
        L96_model.py implementation). Default is False (per-block loop).

    Notes
    -----
    Use ``method='l96_rk4'`` with ``kwargs={'dt': dt, 'si': si}`` to reproduce
    the fixed-step RK4 integration from the reference implementation. This is
    the only solver that correctly captures the two-timescale coupling: adaptive
    solvers (RK45) corrupt ``state_variables`` via sub-step ``dydt`` evaluations,
    and plain Euler at the X timescale is too coarse to resolve the Y dynamics.
    """

    def __init__(self, forcing=None, var_name='lorenz96_two_scale', K=36, J=10,
                 F=10.0, h=1.0, b=10.0, c=10.0, exact_rhs=False,
                 state_variables=None, diagnostic_variables=None, *args, **kwargs):
        if state_variables is None:
            x_names = [f'x{k}' for k in range(K)]
            y_names = [f'y{j}' for j in range(K * J)]
            state_variables = x_names + y_names
        if diagnostic_variables is None:
            diagnostic_variables = []

        # n=K so self.n tracks the slow-scale dimension; full state is K + K*J
        super().__init__(forcing=forcing, var_name=var_name, n=K, F=F,
                         state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables,
                         *args, **kwargs)

        self.K = K
        self.J = J
        self.h = h
        self.b = b
        self.c = c
        self.exact_rhs = exact_rhs
        self.param_values.update({'h': h, 'b': b, 'c': c})

    def _split_state(self, x):
        x = np.asarray(x, dtype=float)
        return x[:self.K], x[self.K:]

    # _state_to_arrays was removed — use model.state_variables[field] directly.
    # def _state_to_arrays(self):
    #     state = self.state_variables
    #     X = np.column_stack([state[f'x{i}'] for i in range(self.K)])
    #     Y = np.column_stack([state[f'y{i}'] for i in range(self.K * self.J)])
    #     t = self.time
    #     return X, Y, t

    def dydt(self, t, x):
        X, Y = self._split_state(x)

        F_t = self._forcing_value(t, x)   # inherited from Lorenz96
        h = self.get_param('h', t, x)
        b = self.get_param('b', t, x)
        c = self.get_param('c', t, x)

        K = self.K
        J = self.J

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

        new_row = np.array([tuple(np.concatenate([X, Y]))], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        return np.concatenate([dX, dY]).tolist()

    def integrate(self, t_span=None, y0=None, method='RK45', kwargs=None, run_name=None):
        """Integrate the two-scale system.

        For correct two-timescale behavior use ``method='l96_rk4'`` with
        ``kwargs={'dt': dt, 'si': si}``, where ``dt`` is the internal RK4 step
        and ``si`` is the output sampling interval (must satisfy si >= dt and
        si/dt an integer).

        All other methods are delegated to ``Lorenz96.integrate`` / the standard
        PBModel path, but note that adaptive solvers (RK45) will corrupt
        ``state_variables`` via sub-step ``dydt`` evaluations, and Euler with
        dt at the X timescale is too coarse to resolve Y dynamics.
        """
        if method != 'l96_rk4':
            return super().integrate(
                t_span=t_span, y0=y0, method=method, kwargs=kwargs, run_name=run_name
            )

        if t_span is None or y0 is None:
            raise ValueError("t_span and y0 must be provided for l96_rk4 integration.")

        kwargs = kwargs or {}
        if 'dt' not in kwargs or 'si' not in kwargs:
            raise ValueError("kwargs must include 'dt' and 'si' for l96_rk4 integration.")

        dt = float(kwargs['dt'])
        si = float(kwargs['si'])
        t0, t1 = float(t_span[0]), float(t_span[1])
        total_time = t1 - t0

        if total_time <= 0:
            raise ValueError("t_span must have t_span[1] > t_span[0].")

        nt = int(round(total_time / si))
        if abs(nt * si - total_time) > 1e-12:
            raise ValueError("t_span length must be an integer multiple of si.")

        if si < dt:
            dt = si
            ns = 1
        else:
            ns = int(round(si / dt))
            if abs(ns * dt - si) > 1e-12:
                raise ValueError("si must be an integer multiple of dt.")

        self.t_span = t_span
        self.y0 = y0
        self.method = method
        self.kwargs = kwargs

        if len(self.state_variables_names) > 0:
            dtype = [(var, float) for var in self.state_variables_names]
            self.dtypes = dtype

        y0_arr = np.asarray(y0, dtype=float)
        history = np.zeros((nt + 1, y0_arr.size), dtype=float)
        time = np.zeros(nt + 1, dtype=float)

        history[0] = y0_arr
        time[0] = t0

        X, Y = self._split_state(y0_arr)

        def rhs(t, X_in, Y_in):
            F_t = self._forcing_value(t, np.concatenate([X_in, Y_in]))
            h = self.get_param('h', t, X_in)
            b = self.get_param('b', t, X_in)
            c = self.get_param('c', t, X_in)

            K = self.K
            J = self.J

            dX = np.zeros(K, dtype=float)
            dY = np.zeros(K * J, dtype=float)

            Y_reshaped = Y_in.reshape(K, J)
            coupling = Y_reshaped.sum(axis=1)

            for k in range(K):
                xm1 = X_in[(k - 1) % K]
                xm2 = X_in[(k - 2) % K]
                xp1 = X_in[(k + 1) % K]
                dX[k] = -xm1 * (xm2 - xp1) - X_in[k] + F_t - (h * c / b) * coupling[k]

            if self.exact_rhs:
                hcb = (h * c) / b
                dY = (-c * b * np.roll(Y_in, -1) *
                      (np.roll(Y_in, -2) - np.roll(Y_in, 1)) -
                      c * Y_in + hcb * np.repeat(X_in, J))
            else:
                for k in range(K):
                    for j in range(J):
                        jm1 = (j - 1) % J
                        jp1 = (j + 1) % J
                        jp2 = (j + 2) % J
                        yjk = Y_reshaped[k, j]
                        dY[k * J + j] = (-c * b * Y_reshaped[k, jp1] *
                                         (Y_reshaped[k, jp2] - Y_reshaped[k, jm1]) -
                                         c * yjk + (h * c / b) * X_in[k])

            return dX, dY

        for n in range(nt):
            base_t = t0 + n * si
            for s in range(ns):
                t_curr = base_t + s * dt
                dX1, dY1 = rhs(t_curr, X, Y)
                dX2, dY2 = rhs(t_curr + 0.5 * dt, X + 0.5 * dt * dX1, Y + 0.5 * dt * dY1)
                dX3, dY3 = rhs(t_curr + 0.5 * dt, X + 0.5 * dt * dX2, Y + 0.5 * dt * dY2)
                dX4, dY4 = rhs(t_curr + dt, X + dt * dX3, Y + dt * dY3)

                X = X + (dt / 6.0) * ((dX1 + dX4) + 2.0 * (dX2 + dX3))
                Y = Y + (dt / 6.0) * ((dY1 + dY4) + 2.0 * (dY2 + dY3))

            history[n + 1] = np.concatenate([X, Y])
            time[n + 1] = t0 + (n + 1) * si

        state = np.zeros(nt + 1, dtype=self.dtypes)
        for i, var in enumerate(self.state_variables_names):
            state[var] = history[:, i]

        self.state_variables = state
        self.time = time
        self.solution = Solution(time, history)
        self.run_name = run_name if run_name is not None else f'{self.method}, dt={dt}, si={si}'
