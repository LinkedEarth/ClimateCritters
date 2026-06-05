from __future__ import annotations

import numpy as np

from ..core.pbmodel import PBModel


class DampedSpring(PBModel):
    """Damped (and optionally driven) spring-mass oscillator.

    Parameters
    ----------
    var_name:
        Human-readable label for the model.
    m:
        Mass in kg.  Must be > 0.
    k:
        Spring constant in N/m.  Must be > 0.
    c:
        Linear damping coefficient in N·s/m.  ``c=0`` gives undamped SHM;
        ``c < 2*sqrt(k*m)`` gives underdamped (oscillatory) decay.
    F : float or callable or pb.core.Forcing
        External driving force (N).  Default 0.0 (undriven).  For a
        time-varying drive use ``model.register_forcing('F', forcing_obj)``.
    state_variables:
        Names for the two integrated state variables (position, velocity).
    diagnostic_variables:
        Names for post-integration diagnostics.

    Notes
    -----
    State variables are ``x`` (position) and ``v`` (velocity) in that order.
    Diagnostic variables ``energy`` and ``omega_0`` are computed after
    integration.

    Examples
    --------

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import paleobeasts as pb
    from paleobeasts.signal_models.damped_spring import DampedSpring

    model = DampedSpring(m=1.0, k=4.0, c=0.4)
    output = model.integrate(t_span=(0, 30), y0=[1.0, 0.0], method='RK45')
    ts = output.to_pyleo(var_names=['x'])
    ts.plot()
    plt.savefig('docs/reference/figures/DampedSpring_example.png',
                dpi=150, bbox_inches='tight')
    ```

    With resonant driving (external force at ω₀):

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import paleobeasts as pb
    from paleobeasts.signal_models.damped_spring import DampedSpring

    m, k = 1.0, 4.0
    omega_0 = np.sqrt(k / m)
    model = DampedSpring(m=m, k=k, c=0.0)
    model.register_forcing('F', pb.core.Forcing(lambda t: np.cos(omega_0 * t)))
    output = model.integrate(t_span=(0, 30), y0=[0.0, 0.0], method='RK45')
    fig, ax = plt.subplots()
    ax.plot(output.time, output.state_variables['x'])
    ax.set_xlabel('time'); ax.set_ylabel('x')
    ax.set_title('DampedSpring — resonant driving at ω₀')
    plt.savefig('docs/reference/figures/DampedSpring_resonance_example.png',
                dpi=150, bbox_inches='tight')
    ```

    """

    def __init__(
        self,
        var_name="damped_spring",
        m=1.0,
        k=1.0,
        c=0.1,
        F=0.0,
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
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.m = m
        self.k = k
        self.c = c
        self.F = F
        self.param_values = {
            "m": m,
            "k": k,
            "c": c,
            "F": F,
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

        F = float(self.get_param_value("F", t, state))

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
