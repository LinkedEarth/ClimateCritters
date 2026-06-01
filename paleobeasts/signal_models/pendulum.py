"""Pendulum signal models.

Three classes covering the main pendulum systems used in introductory dynamics:

- SimplePendulum  : nonlinear damped pendulum (dimensional; L and g explicit)
- DrivenPendulum  : driven damped pendulum — canonical 1-D chaos testbed
- DoublePendulum  : double pendulum — conservative chaotic system

Equations
---------
SimplePendulum (per unit mass, angle in radians):

    dθ/dt = ω
    dω/dt = −λω − ω₀² sin θ       ω₀ = √(g/L)

DrivenPendulum (dimensionless standard form, g = L = m = 1):

    dθ/dt = ω
    dω/dt = −q ω − sin θ + A cos(Ω t)

DoublePendulum (dimensional, no damping by default):

    dθ₁/dt = ω₁
    dθ₂/dt = ω₂
    dω₁/dt = [−g(2m₁+m₂)sin θ₁ − m₂g sin(θ₁−2θ₂)
               − 2 sin(Δ) m₂ (ω₂²L₂ + ω₁²L₁ cos Δ)] / [L₁ D] − d₁ω₁
    dω₂/dt = [2 sin(Δ) (ω₁²L₁(m₁+m₂) + g(m₁+m₂)cos θ₁
               + ω₂²L₂m₂ cos Δ)] / [L₂ D] − d₂ω₂

    where Δ = θ₁ − θ₂ and D = 2m₁ + m₂ − m₂ cos(2Δ).
"""

from __future__ import annotations

import numpy as np

from ..core.pbmodel import PBModel


# ---------------------------------------------------------------------------
# SimplePendulum
# ---------------------------------------------------------------------------

class SimplePendulum(PBModel):
    """Nonlinear pendulum with optional linear damping.

    Parameters
    ----------
    forcing:
        Optional external torque applied additively to ``dω/dt``.  A scalar
        ``Forcing`` value is added directly; if ``None``, the unforced
        dynamics are used.
    var_name:
        Human-readable label.
    L:
        Pendulum length in metres.  Must be > 0.
    g:
        Gravitational acceleration in m/s².  Must be > 0.
    damping:
        Linear damping coefficient λ (s⁻¹).  ``damping=0`` gives undamped SHM.
    state_variables:
        Names for [angle, angular velocity].
    diagnostic_variables:
        Names for post-integration diagnostics.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    from paleobeasts.signal_models.pendulum import SimplePendulum

    model = SimplePendulum(L=1.0, g=9.81, damping=0.1)
    output = model.integrate(t_span=(0, 20), y0=[1.5, 0.0], method='RK45')
    ts = output.to_pyleo(var_names=['theta'])
    ts.plot()
    plt.savefig('docs/reference/figures/SimplePendulum_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(
        self,
        var_name="simple_pendulum",
        L=1.0,
        g=9.81,
        damping=0.0,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["theta", "omega"]
        if diagnostic_variables is None:
            diagnostic_variables = ["energy", "omega_0"]

        super().__init__(
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.L = L
        self.g = g
        self.damping = damping
        self.param_values = {"L": L, "g": g, "damping": damping}
        self.params = ()

    def uses_post_history(self):
        return True

    def dydt(self, t, x):
        state = np.asarray(x, dtype=float).reshape(-1)
        theta, omega = float(state[0]), float(state[1])

        L = float(self.get_param_value("L", t, state))
        g = float(self.get_param_value("g", t, state))
        lam = float(self.get_param_value("damping", t, state))
        if L <= 0.0:
            raise ValueError("L must be > 0.")
        if g <= 0.0:
            raise ValueError("g must be > 0.")

        omega0_sq = g / L
        dtheta = omega
        domega = -lam * omega - omega0_sq * np.sin(theta)
        return [dtheta, domega]

    def populate_diagnostics_from_history(self, time, history):
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)

        energy_vals = np.empty(len(time))
        omega0_vals = np.empty(len(time))

        for i, (t, row) in enumerate(zip(time, history)):
            theta, omega = float(row[0]), float(row[1])
            L = float(self.get_param_value("L", t, row))
            g = float(self.get_param_value("g", t, row))
            energy_vals[i] = 0.5 * (L * omega) ** 2 + g * L * (1.0 - np.cos(theta))
            omega0_vals[i] = np.sqrt(g / L)

        self.diagnostic_variables = {
            "energy": energy_vals,
            "omega_0": omega0_vals,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def natural_frequency(self):
        """Return ω₀ = √(g/L) in rad/s."""
        return np.sqrt(self.param_values["g"] / self.param_values["L"])

    def natural_period(self):
        """Return T₀ = 2π/ω₀ in seconds (small-angle approximation)."""
        return 2.0 * np.pi / self.natural_frequency()

    def damping_ratio(self):
        """Return ζ = λ / (2ω₀).  ζ < 1 underdamped, ζ = 1 critical."""
        return self.param_values["damping"] / (2.0 * self.natural_frequency())


# ---------------------------------------------------------------------------
# DrivenPendulum
# ---------------------------------------------------------------------------

class DrivenPendulum(PBModel):
    """Driven damped pendulum in dimensionless form.

    The standard dimensionless equation (g = L = m = 1):

        dθ/dt = ω
        dω/dt = −q ω − sin θ + A cos(Ω t)

    This system exhibits a rich bifurcation structure: periodic orbits at
    small A, period-doubling, and chaotic strange attractors as A increases.

    Parameters
    ----------
    forcing:
        Optional external drive.  If provided, ``forcing.get_forcing(t)``
        replaces the built-in cosine term entirely, allowing arbitrary drive
        waveforms.  When ``None``, the cosine drive ``A cos(Ω t)`` is used.
    var_name:
        Human-readable label.
    q:
        Damping coefficient (dimensionless).
    A:
        Driving amplitude (dimensionless).  Ignored when ``forcing`` is set.
    Omega:
        Driving angular frequency (dimensionless rad/time).  Ignored when
        ``forcing`` is set.
    state_variables:
        Names for [angle, angular velocity].
    diagnostic_variables:
        Names for post-integration diagnostics.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    from paleobeasts.signal_models.pendulum import DrivenPendulum

    model = DrivenPendulum(q=0.5, A=1.2, Omega=2.0/3.0)
    output = model.integrate(
        t_span=(0, 500), y0=[0.0, 0.0], method='RK45',
        kwargs={'rtol': 1e-9, 'atol': 1e-11},
    )
    theta = output.state_variables['theta']
    omega = output.state_variables['omega']
    fig, ax = plt.subplots()
    ax.plot(theta, omega, ',', ms=0.4, alpha=0.5)
    ax.set_xlabel('θ (rad)'); ax.set_ylabel('ω')
    plt.savefig('docs/reference/figures/DrivenPendulum_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(
        self,
        var_name="driven_pendulum",
        q=0.5,
        A=1.2,
        Omega=2.0 / 3.0,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["theta", "omega"]
        if diagnostic_variables is None:
            diagnostic_variables = ["energy", "drive"]

        super().__init__(
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.q = q
        self.A = A
        self.Omega = Omega
        self.param_values = {"q": q, "A": A, "Omega": Omega}
        self.params = ()

    def uses_post_history(self):
        return True

    def _drive(self, t):
        A = float(self.param_values["A"])
        Omega = float(self.param_values["Omega"])
        return float(A * np.cos(Omega * t))

    def dydt(self, t, x):
        state = np.asarray(x, dtype=float).reshape(-1)
        theta, omega = float(state[0]), float(state[1])

        q = float(self.get_param_value("q", t, state))
        drive = self._drive(t)

        dtheta = omega
        domega = -q * omega - np.sin(theta) + drive
        return [dtheta, domega]

    def populate_diagnostics_from_history(self, time, history):
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)

        energy_vals = np.empty(len(time))
        drive_vals = np.empty(len(time))

        for i, (t, row) in enumerate(zip(time, history)):
            theta, omega = float(row[0]), float(row[1])
            energy_vals[i] = 0.5 * omega ** 2 + (1.0 - np.cos(theta))
            drive_vals[i] = self._drive(t)

        self.diagnostic_variables = {
            "energy": energy_vals,
            "drive": drive_vals,
        }

    def driving_period(self):
        """Return the driving period T_drive = 2π/Ω."""
        return 2.0 * np.pi / self.param_values["Omega"]


# ---------------------------------------------------------------------------
# DoublePendulum
# ---------------------------------------------------------------------------

class DoublePendulum(PBModel):
    """Double pendulum — a conservative chaotic system.

    Two point masses connected by rigid, massless rods swing freely from a
    pivot.  The system is Hamiltonian (no damping by default) and is chaotic
    for most non-trivial initial conditions.

    State variables: [θ₁, ω₁, θ₂, ω₂]

    Parameters
    ----------
    forcing:
        Optional external torque applied additively to ``dω₁/dt`` (the first
        bob).  A scalar ``Forcing`` value is added directly; if ``None``, the
        unforced dynamics are used.
    var_name:
        Human-readable label.
    m1, m2:
        Masses of the two bobs in kg.
    L1, L2:
        Rod lengths in metres.
    g:
        Gravitational acceleration in m/s².
    d1, d2:
        Optional linear damping coefficients on ω₁ and ω₂ (s⁻¹).
        Default 0 (Hamiltonian / energy-conserving).

    Notes
    -----
    The double pendulum is chaotic.  RK45 with default tolerances can show
    significant energy drift for long or large-amplitude runs.  Use tight
    tolerances for accurate energy tracking::

        model.integrate(..., kwargs={'rtol': 1e-10, 'atol': 1e-12})

    state_variables:
        Names for [θ₁, ω₁, θ₂, ω₂].
    diagnostic_variables:
        Names for post-integration diagnostics.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    from paleobeasts.signal_models.pendulum import DoublePendulum

    model = DoublePendulum(m1=1.0, m2=1.0, L1=1.0, L2=1.0)
    output = model.integrate(
        t_span=(0, 60), y0=[np.pi/2, 0.0, np.pi/4, 0.0], method='RK45',
        kwargs={'rtol': 1e-10, 'atol': 1e-12},
    )
    ts = output.to_pyleo(var_names=['theta1'])
    ts.plot()
    plt.savefig('docs/reference/figures/DoublePendulum_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(
        self,
        var_name="double_pendulum",
        m1=1.0,
        m2=1.0,
        L1=1.0,
        L2=1.0,
        g=9.81,
        d1=0.0,
        d2=0.0,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["theta1", "omega1", "theta2", "omega2"]
        if diagnostic_variables is None:
            diagnostic_variables = ["energy", "x1", "y1", "x2", "y2"]

        super().__init__(
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g
        self.d1 = d1
        self.d2 = d2
        self.param_values = {
            "m1": m1, "m2": m2, "L1": L1, "L2": L2,
            "g": g, "d1": d1, "d2": d2,
        }
        self.params = ()

    def uses_post_history(self):
        return True

    def dydt(self, t, x):
        state = np.asarray(x, dtype=float).reshape(-1)
        theta1, omega1, theta2, omega2 = [float(v) for v in state]

        m1 = float(self.get_param_value("m1", t, state))
        m2 = float(self.get_param_value("m2", t, state))
        L1 = float(self.get_param_value("L1", t, state))
        L2 = float(self.get_param_value("L2", t, state))
        g  = float(self.get_param_value("g", t, state))
        d1 = float(self.get_param_value("d1", t, state))
        d2 = float(self.get_param_value("d2", t, state))

        delta = theta1 - theta2
        sin_d = np.sin(delta)
        cos_d = np.cos(delta)
        denom = L1 * (2.0 * m1 + m2 - m2 * np.cos(2.0 * delta))

        domega1 = (
            -g * (2.0 * m1 + m2) * np.sin(theta1)
            - m2 * g * np.sin(theta1 - 2.0 * theta2)
            - 2.0 * sin_d * m2 * (omega2 ** 2 * L2 + omega1 ** 2 * L1 * cos_d)
        ) / denom - d1 * omega1

        domega2 = (
            2.0 * sin_d * (
                omega1 ** 2 * L1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(theta1)
                + omega2 ** 2 * L2 * m2 * cos_d
            )
        ) / (L2 * (2.0 * m1 + m2 - m2 * np.cos(2.0 * delta))) - d2 * omega2

        return [omega1, domega1, omega2, domega2]

    def populate_diagnostics_from_history(self, time, history):
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)

        n = len(time)
        energy = np.empty(n)
        x1 = np.empty(n)
        y1 = np.empty(n)
        x2 = np.empty(n)
        y2 = np.empty(n)

        m1 = self.param_values["m1"]
        m2 = self.param_values["m2"]
        L1 = self.param_values["L1"]
        L2 = self.param_values["L2"]
        g  = self.param_values["g"]

        for i, (t, row) in enumerate(zip(time, history)):
            th1, om1, th2, om2 = [float(v) for v in row]
            delta = th1 - th2

            # Kinetic energy
            KE = (
                0.5 * m1 * (L1 * om1) ** 2
                + 0.5 * m2 * (
                    (L1 * om1) ** 2
                    + (L2 * om2) ** 2
                    + 2.0 * L1 * L2 * om1 * om2 * np.cos(delta)
                )
            )
            # Potential energy (zero at pivot)
            PE = (
                -(m1 + m2) * g * L1 * np.cos(th1)
                - m2 * g * L2 * np.cos(th2)
            )
            energy[i] = KE + PE

            # Cartesian positions (y positive upward, origin at pivot)
            x1[i] =  L1 * np.sin(th1)
            y1[i] = -L1 * np.cos(th1)
            x2[i] =  x1[i] + L2 * np.sin(th2)
            y2[i] =  y1[i] - L2 * np.cos(th2)

        self.diagnostic_variables = {
            "energy": energy,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
        }

    def cartesian_positions(self):
        """Return (x1, y1, x2, y2) from the last integration run."""
        dv = self.diagnostic_variables
        return dv["x1"], dv["y1"], dv["x2"], dv["y2"]
