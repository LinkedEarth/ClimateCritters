"""Damped (and optionally driven) spring-mass oscillator.

Models the 1-D motion of a mass on a spring with linear damping:

    dx/dt = v
    dv/dt = -(c/m)*v - (k/m)*x + F(t)/m

where
    x  — displacement from equilibrium  [m]
    v  — velocity                        [m/s]
    m  — mass                            [kg]
    k  — spring constant                 [N/m]
    c  — damping coefficient             [N·s/m]
    F(t) — optional external driving force [N]; provided via ``forcing``

With no forcing the system is autonomous (damped SHM).  Providing a
``Forcing`` object enables the driven case, which exhibits resonance at
the natural frequency ω₀ = √(k/m).

Fixed point: (x*, v*) = (0, 0) — stable when c > 0.

Natural frequency and period:
    ω₀ = √(k/m)          [rad/s]
    T₀ = 2π/ω₀           [s]

Diagnostic variables (computed post-integration):
    energy  — total mechanical energy  ½mv² + ½kx²  [J]
    omega_0 — natural angular frequency              [rad/s]
"""

from __future__ import annotations

import numpy as np

from ..core.pbmodel import PBModel


class DampedSpring(PBModel):
    """Damped (and optionally driven) spring-mass oscillator.

    Parameters
    ----------
    forcing:
        Optional external driving force F(t).  Pass a ``Forcing`` object;
        ``get_forcing(t)`` should return force in Newtons.  If ``None``,
        the undriven damped oscillator is simulated.
    var_name:
        Human-readable label for the model.
    m:
        Mass in kg.  Must be > 0.
    k:
        Spring constant in N/m.  Must be > 0.
    c:
        Linear damping coefficient in N·s/m.  ``c=0`` gives undamped SHM;
        ``c < 2*sqrt(k*m)`` gives underdamped (oscillatory) decay.
    state_variables:
        Names for the two integrated state variables (position, velocity).
    diagnostic_variables:
        Names for post-integration diagnostics.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    from paleobeasts.signal_models.damped_spring import DampedSpring

    model = DampedSpring(forcing=None, m=1.0, k=4.0, c=0.4)
    output = model.integrate(t_span=(0, 30), y0=[1.0, 0.0], method='RK45')
    ts = output.to_pyleo(var_names=['x'])
    ts.plot()
    plt.savefig('docs/reference/figures/DampedSpring_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(
        self,
        forcing=None,
        var_name="damped_spring",
        m=1.0,
        k=1.0,
        c=0.1,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["x", "v"]
        if diagnostic_variables is None:
            diagnostic_variables = ["energy", "omega_0"]

        super().__init__(
            forcing,
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.m = m
        self.k = k
        self.c = c
        self.param_values = {
            "m": m,
            "k": k,
            "c": c,
        }
        self.params = ()

    # ------------------------------------------------------------------
    # PBModel interface
    # ------------------------------------------------------------------

    def uses_post_history(self):
        return True

    def dydt(self, t, x):
        state = np.asarray(x, dtype=float).reshape(-1)
        pos, vel = float(state[0]), float(state[1])

        m = float(self.get_param_value("m", t, state))
        k = float(self.get_param_value("k", t, state))
        c = float(self.get_param_value("c", t, state))
        if m <= 0.0:
            raise ValueError("m must be > 0.")
        if k <= 0.0:
            raise ValueError("k must be > 0.")

        F = float(self.resolve_forcing(t))

        dxdt = vel
        dvdt = -(c / m) * vel - (k / m) * pos + F / m
        return [dxdt, dvdt]

    def populate_diagnostics_from_history(self, time, history):
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)

        energy_vals = np.empty(len(time))
        omega0_vals = np.empty(len(time))

        for i, (t, row) in enumerate(zip(time, history)):
            pos, vel = float(row[0]), float(row[1])
            m = float(self.get_param_value("m", t, row))
            k = float(self.get_param_value("k", t, row))
            energy_vals[i] = 0.5 * m * vel ** 2 + 0.5 * k * pos ** 2
            omega0_vals[i] = np.sqrt(k / m)

        self.diagnostic_variables = {
            "energy": energy_vals,
            "omega_0": omega0_vals,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def natural_frequency(self):
        """Return ω₀ = √(k/m) in rad/s (uses current param_values)."""
        return np.sqrt(self.param_values["k"] / self.param_values["m"])

    def natural_period(self):
        """Return T₀ = 2π/ω₀ in seconds (uses current param_values)."""
        return 2.0 * np.pi / self.natural_frequency()

    def damping_ratio(self):
        """Return ζ = c / (2√(km)).  ζ<1 underdamped, ζ=1 critical, ζ>1 overdamped."""
        m = self.param_values["m"]
        k = self.param_values["k"]
        c = self.param_values["c"]
        return c / (2.0 * np.sqrt(k * m))
