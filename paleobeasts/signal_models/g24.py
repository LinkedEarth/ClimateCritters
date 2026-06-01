import numpy as np
from ..core.pbmodel import PBModel
from scipy.interpolate import CubicSpline


class Model3(PBModel):
    """Model 3 from Ganopolski (2024) describing glacial cycle evolution under orbital forcing.

    The model tracks ice volume ``v`` and glacial regime ``k`` (1 = glaciation,
    2 = deglaciation).  The ice volume relaxes toward a forcing-dependent
    equilibrium:

        dv/dt = (ve(f) - v) / t1    when k=1  (glaciation)
        dv/dt = -vc / t2             when k=2  (deglaciation)

    Regime switches follow:

    - k=1 → k=2  if v > vc, df/dt > 0, and f > 0
    - k=2 → k=1  if f < f1

    Parameters
    ----------
    var_name : str
        Label for the model output.  Default ``'ice volume'``.
    f1 : float or callable or pb.core.Forcing
        Insolation threshold for glacial inception (W m\ :sup:`-2`;
        typically -20 to -15).  Default -16.
    f2 : float or callable or pb.core.Forcing
        Insolation threshold for deglaciation inception (W m\ :sup:`-2`;
        positive).  Default 16.
    t1 : float or callable or pb.core.Forcing
        Relaxation timescale for glacial inception (kyr).  Default 30.
    t2 : float or callable or pb.core.Forcing
        Relaxation timescale for deglaciation (kyr).  Default 10.
    vc : float or callable or pb.core.Forcing
        Critical ice volume controlling dominant periodicity and asymmetry.
        Default 1.4.

    Notes
    -----
    State variables are ``v`` (ice volume, normalised) and ``k`` (regime
    index, non-integrated).  The diagnostic variable ``insolation`` records
    ``f(t)`` at each output step.

    The ``insolation`` parameter (default 0.0) provides the orbital forcing
    value used by the model.  Register a time-varying orbital signal via::

        model.register_forcing('insolation', pb.core.Forcing(calc_f))

    The internal derivative of forcing ``dfdt`` is computed via
    :func:`calc_df` by default; supply a custom callable or
    ``pb.core.Forcing`` to override.

    Parameter defaults are taken from Ganopolski (2024).  Time-varying
    parameters follow the standard callable contract ``(t)``,
    ``(t, state)``, or ``(t, state, model)``.

    References
    ----------
    Ganopolski, A. (2024). Glacial cycles. *Nature Reviews Earth &
    Environment*, 5, 89–106.

    Examples
    --------
    ```python
    import paleobeasts as pb
    from paleobeasts.signal_models.g24 import Model3, calc_f
    import matplotlib.pyplot as plt

    model = Model3()
    model.register_forcing('insolation', pb.core.Forcing(calc_f))

    output = model.integrate(
        t_span=(-2000, 0), y0=[0.0, 1], method='RK45',
        kwargs={'max_step': 0.5}
    )
    ts = output.to_pyleo(var_names=['v'])
    ts.plot()
    plt.savefig('docs/reference/figures/Model3_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """

    def __init__(self, var_name='ice volume', f1=-16, f2=16, t1=30, t2=10, vc=1.4,
                 insolation=0.0,
                 state_variables=['v', 'k'], non_integrated_state_vars=['k'], diagnostic_variables=['insolation'], *args,
                 **kwargs):
        super().__init__(var_name, state_variables=state_variables,
                         non_integrated_state_vars=non_integrated_state_vars,
                         diagnostic_variables=diagnostic_variables, *args, **kwargs)
        self.f1 = f1
        self.f2 = f2
        self.t1 = t1
        self.t2 = t2
        self.vc = vc
        self.insolation = insolation
        self.dfdt = calc_df
        self.param_values = {
            'f1': f1,
            'f2': f2,
            't1': t1,
            't2': t2,
            'vc': vc,
            'dfdt': self.dfdt,
            'insolation': insolation,
        }
        self.params = ()

    def dydt(self, t, x):
        """Evaluate ice-volume tendency at time ``t`` and state ``x``.

        Returns
        -------
        list of float
            ``[dvdt]`` — the ice-volume tendency.  The regime index ``k``
            is a non-integrated state variable updated in-place.
        """
        v = x[0]  # int(self.state_variables[-1][0])
        if isinstance(v, np.ndarray):
            v = v[-1]

        f1 = self.get_param_value('f1', t, x)
        f2 = self.get_param_value('f2', t, x)
        t1 = self.get_param_value('t1', t, x)
        t2 = self.get_param_value('t2', t, x)

        k = int(self.state_variables['k'][-1])
        f = self.get_param_value('insolation', t, x)
        dfdt = self.calc_dfdt(self.time_util(t), x)

        vc = self.get_param_value('vc', t, x)

        k = self.calc_k(k, dfdt, f, v, vc, f1)

        if k == 1:
            ve = self.calc_ve(v, f, f1, f2)
            dvdt = (ve - v) / t1
        elif k == 2:
            dvdt = -vc / t2

        if t>0:
            new_row = np.array([(v, k)], dtype=self.dtypes)
            self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
            self.time.append(t)
        self.diagnostic_variables['insolation'].append(f)

        return [dvdt]

    def calc_k(self, k, dfdt, f, v, vc, f1):
        if k == 1 and dfdt > 0 and f > 0 and v > vc:
            k = 2
        elif k == 2 and f < f1:
            k = 1
        return k

    def calc_ve(self, v, f, f1, f2, vi=0):
        """Equilibrium ice volume toward which the system is attracted.

        In the bi-stable regime (``f1 < f < f2``) the target depends on the
        current ice volume ``v`` relative to the unstable equilibrium ``vu``.

        Parameters
        ----------
        v : float
            Current ice volume (normalised).
        f : float
            Orbital forcing value (W m\ :sup:`-2`).
        f1 : float
            Lower insolation threshold (glaciation onset).
        f2 : float
            Upper insolation threshold (deglaciation onset).
        vi : float
            Interglacial equilibrium ice volume.  Default 0.

        Returns
        -------
        ve : float
            Target equilibrium ice volume.
        """
        vg = self.calc_vg(f, f1, f2)
        vu = self.calc_vu(f, f1, f2)

        if f < f1:
            return vg
        elif f > f2:
            return vi
        elif f1 < f < f2 and v > vu:
            return vg
        elif f1 < f < f2 and v < vu:
            return vi

    def calc_vg(self, f, f1, f2):
        """Glacial equilibrium ice volume.

        Parameters
        ----------
        f : float
            Orbital forcing value (W m\ :sup:`-2`).
        f1 : float
            Lower insolation threshold.
        f2 : float
            Upper insolation threshold.

        Returns
        -------
        vg : float
            Glacial equilibrium ice volume.
        """
        return 1 + np.sqrt((f2 - f) / (f2 - f1))

    def calc_vu(self, f, f1, f2):
        """Unstable equilibrium ice volume separating glacial and interglacial basins.

        Parameters
        ----------
        f : float
            Orbital forcing value (W m\ :sup:`-2`).
        f1 : float
            Lower insolation threshold.
        f2 : float
            Upper insolation threshold.

        Returns
        -------
        vu : float
            Unstable equilibrium ice volume.
        """
        return 1 - np.sqrt((f2 - f) / (f2 - f1))

    def calc_vc(self, t, x=None):
        """Evaluate the critical ice volume ``vc`` (constant or time/state-varying)."""
        if x is None:
            x = self.state_variables[-1] if self.state_variables is not None else None
        return self.get_param_value('vc', t, x)

    def calc_dfdt(self, t, x=None):
        """Evaluate df/dt — the time derivative of the orbital forcing at time ``t``."""
        if x is None:
            x = self.state_variables[-1] if self.state_variables is not None else None
        dfdt_spec = self.param_values['dfdt']
        eval_t = self.time_util(t)

        if hasattr(dfdt_spec, 'get_forcing'):
            return dfdt_spec.get_forcing(eval_t)
        if callable(dfdt_spec):
            # Prefer time-only evaluation first for scientific derivative functions
            # like calc_df(t, A=..., eps=..., T1=..., T2=...) used in the paper.
            try:
                return dfdt_spec(eval_t)
            except TypeError:
                return self.resolve_param(dfdt_spec, t, x)
        return dfdt_spec


def calc_df(t, A=25, eps=0.5, T1=100, T2=30):
    """Derivative of the Ganopolski (2024) orbital forcing at time ``t``.

    Parameters
    ----------
    t : float
        Time (kyr).
    A : float
        Forcing amplitude (W m\ :sup:`-2`).  Default 25.
    eps : float
        Eccentricity modulation amplitude.  ``eps=0`` removes eccentricity
        modulation.  Default 0.5.
    T1 : float
        Eccentricity timescale (kyr).  Default 100.
    T2 : float
        Precession timescale (kyr).  Default 30.

    Returns
    -------
    float
        df/dt at time ``t``.
    """
    return A * eps * ((2 * np.pi / T1) * np.cos(2 * np.pi * t / T1) * np.cos(2 * np.pi * t / T2) -
                      (2 * np.pi / T2) * np.sin(2 * np.pi * t / T2) * np.sin(2 * np.pi * t / T1))


def calc_f(t, A=25, eps=0.5, T1=100, T2=30):
    """Ganopolski (2024) orbital forcing value at time ``t``.

    Parameters
    ----------
    t : float
        Time (kyr).
    A : float
        Forcing amplitude (W m\ :sup:`-2`).  Default 25.
    eps : float
        Eccentricity modulation amplitude.  Default 0.5.
    T1 : float
        Eccentricity timescale (kyr).  Default 100.
    T2 : float
        Precession timescale (kyr).  Default 30.

    Returns
    -------
    float
        Orbital forcing value at time ``t``.
    """
    return A * (1 + eps * np.sin(2 * np.pi * t / T1)) * np.cos(2 * np.pi * t / T2)


def vc_func(t, vc1=0.65, vc2=1.38, t1_mpt=-1050, tau1=250):
    """Time-varying critical ice volume for the Mid-Pleistocene Transition.

    Returns a sigmoid ramp between ``vc1`` (pre-MPT) and ``vc2`` (post-MPT)
    following Ganopolski (2024):

        vc(t) = 0.5*(vc1 + vc2) + 0.5*(vc2 - vc1) * tanh((t - t1_mpt)/tau1)

    Parameters
    ----------
    t : float
        Time (kyr).
    vc1 : float
        Pre-MPT critical ice volume.  Default 0.65.
    vc2 : float
        Post-MPT critical ice volume.  Default 1.38.
    t1_mpt : float
        Centre of the MPT transition (kyr).  Default -1050.
    tau1 : float
        Width of the MPT transition (kyr).  Default 250.

    Returns
    -------
    float
        Critical ice volume at time ``t``.
    """
    return 0.5 * (vc1 + vc2) + 0.5 * (vc2 - vc1) * np.tanh((t - t1_mpt) / tau1)
