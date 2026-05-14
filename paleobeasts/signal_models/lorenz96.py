import numpy as np

from ..core.pbmodel import PBModel
from ..utils.solver import Solution


class Lorenz96(PBModel):
    """Lorenz (1996) single-scale and two-scale atmospheric model.

    A periodic ring of *n* variables with quadratic advection and constant
    forcing. When *J* > 0 a second, faster layer of *n* × *J* variables is
    coupled to the slow layer (the two-scale system of Lorenz & Emanuel 1998).

    Parameters
    ----------
    forcing : pb.Forcing or None
        Forcing object providing F(t). If None, the constant ``F`` parameter
        is used.
    var_name : str
        Default ``'lorenz96'``.
    n : int
        Number of slow-scale variables. Default 40.
    J : int
        Fast variables per slow variable. ``J=0`` (default) gives the
        single-scale system; ``J>0`` gives the two-scale system.
    F : float or callable or pb.Forcing
        Slow-scale forcing. Default 8.
    h : float
        Coupling coefficient between X and Y (two-scale only). Default 1.
    b : float
        Amplitude ratio Y/X (two-scale only). Default 10.
    c : float
        Timescale ratio Y/X (two-scale only). Default 10.
    exact_rhs : bool
        If True, use global ``np.roll`` on the flattened Y vector, matching
        the original L96_model.py. Default False (per-block loop).

    Notes
    -----
    For the two-scale system (``J>0``) use ``method='l96_rk4'`` with
    ``kwargs={'dt': dt, 'si': si}``. Adaptive solvers (RK45) corrupt
    ``state_variables`` via sub-step ``dydt`` evaluations; plain Euler is
    too coarse to resolve the Y dynamics.

    References
    ----------
    Lorenz (1996). Predictability: A problem partly solved.
    Lorenz & Emanuel (1998). J. Atmos. Sci., 55, 399–414.
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
            return self.get_param('F', t, x)
        return self.forcing.get_forcing(self.time_util(t))

    def dydt(self, t, x):
        x = np.asarray(x, dtype=float)

        if self.J == 0:
            return self._dydt_single(t, x)
        return self._dydt_two_scale(t, x)

    def _dydt_single(self, t, x):
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

    def _dydt_two_scale(self, t, x):
        K, J = self.n, self.J
        X, Y = x[:K], x[K:]

        F_t = self._forcing_value(t, x)
        h = self.get_param('h', t, x)
        b = self.get_param('b', t, x)
        c = self.get_param('c', t, x)

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
        """Integrate the model.

        For the two-scale system (``J>0``) pass ``method='l96_rk4'`` with
        ``kwargs={'dt': dt, 'si': si}``. All other methods delegate to the
        standard PBModel integration path.
        """
        if method == 'l96_rk4' and self.J == 0:
            raise ValueError(
                "method='l96_rk4' is only valid for the two-scale system (J > 0). "
                "For the single-scale system use method='RK45' or method='euler'."
            )

        if self.J == 0 or method != 'l96_rk4':
            return super().integrate(
                t_span=t_span, y0=y0, method=method, kwargs=kwargs,
                run_name=run_name
            )

        # ── l96_rk4: fixed-step RK4 at the Y timescale ────────────────────
        if t_span is None or y0 is None:
            raise ValueError("t_span and y0 must be provided for l96_rk4.")

        kwargs = kwargs or {}
        if 'dt' not in kwargs or 'si' not in kwargs:
            raise ValueError("kwargs must include 'dt' and 'si' for l96_rk4.")

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

        dtype = [(var, float) for var in self.state_variables_names]
        self.dtypes = dtype

        y0_arr = np.asarray(y0, dtype=float)
        history = np.zeros((nt + 1, y0_arr.size), dtype=float)
        time = np.zeros(nt + 1, dtype=float)

        history[0] = y0_arr
        time[0] = t0

        K, J = self.n, self.J
        X, Y = y0_arr[:K].copy(), y0_arr[K:].copy()

        def rhs(t, X_in, Y_in):
            F_t = self._forcing_value(t, np.concatenate([X_in, Y_in]))
            h = self.get_param('h', t, X_in)
            b = self.get_param('b', t, X_in)
            c = self.get_param('c', t, X_in)

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

        for step in range(nt):
            base_t = t0 + step * si
            for s in range(ns):
                t_curr = base_t + s * dt
                dX1, dY1 = rhs(t_curr, X, Y)
                dX2, dY2 = rhs(t_curr + 0.5 * dt, X + 0.5 * dt * dX1, Y + 0.5 * dt * dY1)
                dX3, dY3 = rhs(t_curr + 0.5 * dt, X + 0.5 * dt * dX2, Y + 0.5 * dt * dY2)
                dX4, dY4 = rhs(t_curr + dt, X + dt * dX3, Y + dt * dY3)

                X = X + (dt / 6.0) * ((dX1 + dX4) + 2.0 * (dX2 + dX3))
                Y = Y + (dt / 6.0) * ((dY1 + dY4) + 2.0 * (dY2 + dY3))

            history[step + 1] = np.concatenate([X, Y])
            time[step + 1] = t0 + (step + 1) * si

        state = np.zeros(nt + 1, dtype=self.dtypes)
        for i, var in enumerate(self.state_variables_names):
            state[var] = history[:, i]

        self.state_variables = state
        self.time = time
        self.solution = Solution(time, history)
        self.run_name = run_name if run_name is not None else f'{self.method}, dt={dt}, si={si}'


# =============================================================================
# OLD Lorenz96TwoScale(PBModel) — commented out 2026-05-13
# Replaced first by Lorenz96TwoScale(Lorenz96) (subclass, same date),
# then merged into Lorenz96 via the J parameter (2026-05-13).
# Kept for reference during troubleshooting.
# =============================================================================
#
# class Lorenz96TwoScale(PBModel):
# ... (see git history)
