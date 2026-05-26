import numpy as np

from ..utils import constants as phys
from ..core.pbmodel import PBModel

__all__ = [
    'EBMBase', 'EBM0D', 'EBM1DLat',
    'OLR_func', 'albedo_func', 'albedo_func1D',
]


# ---------------------------------------------------------------------------
# Module-level physics helpers
# All callables comply with the PBModel contract: (t), (t, state), or
# (t, state, model) with the first positional arg named 't' or 'time'.
# ---------------------------------------------------------------------------

def _scalar_albedo(T, alpha_ice=0.6, alpha_0=0.3, T1=260., T2=290.):
    """Ice-albedo feedback for a single temperature value (internal helper)."""
    if T < T1:
        return alpha_ice
    if T <= T2:
        r = (T - T2) ** 2 / (T2 - T1) ** 2
        return alpha_0 + (alpha_ice - alpha_0) * r
    return alpha_0


def _olr_stefan_boltzmann(t, state, pRad=650, ps=1000):
    """Stefan-Boltzmann OLR assuming a dry-adiabatic lapse rate (internal helper)."""
    Ts = float(np.asarray(state).reshape(-1)[0])
    Te = (pRad / ps) ** (2. / 7.) * Ts
    return phys.sigma * Te ** 4.


def OLR_func(pRad=650, ps=1000):
    """Return a compliant ``(t, state)`` callable for Stefan-Boltzmann OLR.

    The returned callable computes outgoing longwave radiation by mapping the
    surface temperature to an effective emission temperature using a dry-adiabatic
    pressure scaling, then applying the Stefan-Boltzmann law.

    Parameters
    ----------
    pRad : float, optional
        Radiative pressure level in hPa.  Default is 650.
    ps : float, optional
        Surface pressure in hPa.  Default is 1000.

    Returns
    -------
    func : callable
        A ``(t, state)`` callable compliant with the PBModel parameter
        contract (see ``contracts/signal_model_contract.md``).  Returns OLR
        in W m⁻².
    """
    def _olr(t, state):
        return _olr_stefan_boltzmann(t, state, pRad, ps)
    return _olr


def albedo_func(t, state, *, alpha_ice=0.6, alpha_0=0.3, T1=260., T2=290.):
    """Temperature-dependent albedo with a smooth ice-line transition.

    Returns a single albedo value based on the global-mean (or scalar)
    temperature.  Fully ice-covered below ``T1``, ice-free above ``T2``,
    with a quadratic blend in between.

    Parameters
    ----------
    t : float
        Current time.  Unused; kept for contract compliance.
    state : float or array-like
        Current temperature.  The first element is used if array-like.
    alpha_ice : float, optional
        Albedo for the cold (ice-covered) state.  Default is 0.6.
    alpha_0 : float, optional
        Albedo for the warm (ice-free) state.  Default is 0.3.
    T1 : float, optional
        Temperature (K) below which full ice albedo applies.  Default is 260.
    T2 : float, optional
        Temperature (K) above which full warm albedo applies.  Default is 290.

    Returns
    -------
    albedo : float
        Planetary albedo in [0, 1].
    """
    Ts = float(np.asarray(state).reshape(-1)[0])
    return _scalar_albedo(Ts, alpha_ice, alpha_0, T1, T2)


def albedo_func1D(t, state, model, *, a2=0.25, alpha_ice=0.6, alpha_0=0.1, T1=260., T2=290.):
    """P2-corrected latitudinal albedo using a Legendre polynomial parameterization.

    Uses the global-mean temperature to set the base albedo via the same
    quadratic ice-line transition as :func:`albedo_func`, then adds a
    second-order Legendre polynomial correction to capture the equator-to-pole
    gradient.  Requires the model to expose a ``phi`` attribute (degrees).

    Parameters
    ----------
    t : float
        Current time.  Unused; kept for contract compliance.
    state : array-like
        Full temperature array of length ``grid_n``.  Global mean is computed
        internally.
    model : EBMBase subclass
        Model instance.  Must have a ``phi`` attribute (latitude in degrees).
    a2 : float, optional
        Amplitude of the P2 Legendre polynomial correction.  Default is 0.25.
    alpha_ice : float, optional
        Albedo for the cold (ice-covered) state.  Default is 0.6.
    alpha_0 : float, optional
        Albedo for the warm (ice-free) state.  Default is 0.1.
    T1 : float, optional
        Temperature (K) below which full ice albedo applies.  Default is 260.
    T2 : float, optional
        Temperature (K) above which full warm albedo applies.  Default is 290.

    Returns
    -------
    albedo : ndarray
        Latitudinal albedo array of length ``grid_n``, one value per grid point.
    """
    phi = model.phi
    Ts = float(np.mean(np.asarray(state, dtype=float)))
    a0 = _scalar_albedo(Ts, alpha_ice, alpha_0, T1, T2)
    P2 = 0.5 * (3 * np.sin(np.deg2rad(phi)) ** 2 - 1)
    return a0 + a2 * P2


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class EBMBase(PBModel):
    """Shared energy balance physics for all EBM variants.

    Provides default implementations of ``calc_OLR``, ``calc_albedo``, and
    ``calc_C`` that delegate to ``param_values`` via ``get_param_value``.
    Subclasses with different formulations (e.g. ``EBM1DLat`` with Budyko
    linear OLR and an array ice-line albedo) override the relevant
    ``calc_*`` methods.  The external signature ``(self, T, t)`` is shared
    across all variants so user code can call these helpers uniformly.

    Notes
    -----
    This class is not meant to be instantiated directly.  Subclass it and
    override at minimum ``dydt`` and, where needed, the ``calc_*`` helpers.

    See also
    --------
    paleobeasts.core.PBModel : Base class for all PaleoBeasts models.
    EBM0D : Zero-dimensional EBM subclass.
    EBM1DLat : Latitudinally-resolved diffusive EBM subclass.
    """

    def calc_OLR(self, T, t):
        """Return OLR at state T and time t (delegates to param_values['OLR'])."""
        return self.get_param_value('OLR', t, T)

    def calc_albedo(self, T, t):
        """Return albedo at state T and time t (delegates to param_values['albedo'])."""
        return self.get_param_value('albedo', t, T)

    def calc_C(self, T, t):
        """Return heat capacity at state T and time t (delegates to param_values['C'])."""
        return self.get_param_value('C', t, T)


# ---------------------------------------------------------------------------
# 0D variant
# ---------------------------------------------------------------------------

class EBM0D(EBMBase):
    """Zero-dimensional energy balance model for global-mean surface temperature.

    Evolves global-mean surface temperature T according to:

        C dT/dt = (1 - alpha) * S0/4 - OLR

    where S0 is the solar constant supplied by ``forcing``, ``alpha`` is the
    planetary albedo, and OLR is the outgoing longwave radiation.

    Parameters
    ----------
    forcing : pb.core.Forcing
        Provides the solar constant S0 (W m⁻²) as a function of time.
    state_variables : list of str, optional
        Names of the integrated state variables.  Default is ``['T']``.
    diagnostic_variables : list of str, optional
        Names of diagnostic quantities accumulated during integration.
        Default is ``['albedo', 'absorbed_SW', 'OLR', 'solar_incoming']``.
    var_name : str, optional
        Label for the modeled quantity.  Default is ``'temperature'``.
    OLR : callable or None, optional
        Outgoing longwave radiation.  Must have a compliant signature:
        ``(t)``, ``(t, state)``, or ``(t, state, model)`` with the first
        argument named ``t`` or ``time`` (see
        ``contracts/signal_model_contract.md``).  Default is
        Stefan-Boltzmann via ``OLR_func(pRad=650, ps=1000)``.
    C : float or callable or pb.core.Forcing, optional
        Heat capacity (W yr m⁻² K⁻¹).  If callable, must follow the
        parameter callable contract.  Default is 4.
    albedo : float or callable or pb.core.Forcing, optional
        Planetary albedo.  If callable, must follow the parameter callable
        contract.  Default is 0.3.

    Notes
    -----
    All parameters that support callables or Forcing objects are registered
    in ``param_values`` (``'C'``, ``'albedo'``, ``'OLR'``) and resolved
    at each timestep via ``get_param_value``.

    This model uses ``uses_post_history = False``: state and diagnostics are
    accumulated step-by-step inside ``dydt``.

    See also
    --------
    EBM1DLat : Latitudinally-resolved 1D variant.
    OLR_func : Default OLR callable factory.
    albedo_func : Ice-albedo callable for use as the ``albedo`` parameter.

    Examples
    --------

    .. code-block:: python

        import paleobeasts as pb
        from paleobeasts.signal_models.ebm import EBM0D

        forcing = pb.core.Forcing(lambda t: 1360.0)
        model = EBM0D(forcing=forcing)
        output = model.integrate(t_span=(0, 500), y0=[288.0], method='RK45')

    With a time-varying albedo and custom OLR:

    .. code-block:: python

        from paleobeasts.signal_models.ebm import EBM0D, albedo_func, OLR_func

        model = EBM0D(
            forcing=forcing,
            albedo=albedo_func,
            OLR=OLR_func(pRad=600),
        )

    """

    def __init__(self, forcing, var_name='temperature', state_variables=None,
                 diagnostic_variables=None, OLR=None, C=4, albedo=0.3):
        if state_variables is None:
            state_variables = ['T']
        if diagnostic_variables is None:
            diagnostic_variables = ['albedo', 'absorbed_SW', 'OLR', 'solar_incoming']

        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables)

        self.C = C
        self.albedo = albedo
        self.OLR = OLR if OLR is not None else OLR_func()
        self.param_values = {
            'C': self.C,
            'albedo': self.albedo,
            'OLR': self.OLR,
        }
        self.params = ()

    def dydt(self, t, x):
        """Evaluate the right-hand side of the ODE at time t and state x.

        Called by the solver at each timestep.  As a side effect, appends the
        current state to ``self.state_variables`` and appends each diagnostic
        quantity to the corresponding list in ``self.diagnostic_variables``.

        Parameters
        ----------
        t : float
            Current time.
        x : array-like
            Current state vector.  ``x[0]`` is the global-mean temperature T (K).

        Returns
        -------
        dydt : list of float
            Time-derivatives ``[dT/dt]``.
        """
        T = float(x[0])

        f_solar_incoming = self.forcing.get_forcing(t)
        albedo = self.calc_albedo(T, t)
        absorbed_SW = (1 - albedo) * f_solar_incoming / 4
        OLR = self.calc_OLR(T, t)
        C = self.calc_C(T, t)
        dTdt = (absorbed_SW - OLR) / C

        new_row = np.array([(T,)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        self.diagnostic_variables['albedo'].append(albedo)
        self.diagnostic_variables['absorbed_SW'].append(absorbed_SW)
        self.diagnostic_variables['OLR'].append(OLR)
        self.diagnostic_variables['solar_incoming'].append(f_solar_incoming)

        if t > 0:
            self.time.append(t)

        return [dTdt]


# ---------------------------------------------------------------------------
# 1D latitudinal variant
# ---------------------------------------------------------------------------

class EBM1DLat(EBMBase):
    """Diffusive annual-mean latitudinal energy balance model (Budyko-Sellers type).

    Evolves the zonal-mean temperature profile T(phi) on a latitude grid:

        C dT/dt = S(x)(1 - alpha(T)) - OLR(T) + D * div(grad T)

    where ``x = sin(phi)``, OLR follows the Budyko linear form
    ``(A - CO2_forcing) + B * T``, and diffusion is computed in x-coordinates
    with no-flux polar boundary conditions.

    Parameters
    ----------
    forcing : pb.core.Forcing or None, optional
        Reserved for future use (e.g. external orbital forcing).
        Default is None; solar input is controlled by ``S0`` instead.
    var_name : str, optional
        Label for the modeled quantity.  Default is ``'ebm1d_lat'``.
    grid_n : int, optional
        Number of evenly-spaced latitude grid points from -90° to 90°.
        Must be ≥ 3.  Default is 50.
    C : float or callable or pb.core.Forcing, optional
        Heat capacity (W yr m⁻² K⁻¹).  Default is 10.0.
    D : float or callable or pb.core.Forcing, optional
        Meridional diffusion coefficient.  Default is 0.55.
    A : float or callable or pb.core.Forcing, optional
        Budyko OLR intercept (W m⁻²).  Default is 210.0.
    B : float or callable or pb.core.Forcing, optional
        Budyko OLR slope (W m⁻² K⁻¹).  Default is 2.0.
    S0 : float or callable or pb.core.Forcing, optional
        Solar constant (W m⁻²).  Default is 1365.0.
    CO2_forcing : float or callable or pb.core.Forcing, optional
        Radiative forcing from CO2 (W m⁻²); shifts the OLR intercept down,
        warming the climate.  Default is 0.0.
    state_variables : list of str, optional
        Names of the integrated state variables.  Defaults to
        ``['T_0', 'T_1', ..., 'T_{grid_n-1}']``.
    diagnostic_variables : list of str, optional
        Names of diagnostic quantities computed from the solved trajectory.
        Default is ``['ice_line_lat', 'Tglobal']``.

    Notes
    -----
    ``uses_post_history = True``: state variables and diagnostics are derived
    from the full solved trajectory in ``populate_diagnostics_from_history``
    rather than accumulated step-by-step in ``dydt``.

    ``validate_initial_state`` accepts a scalar and broadcasts it uniformly
    to the full grid.

    All parameters (``C``, ``D``, ``A``, ``B``, ``S0``, ``CO2_forcing``) are
    registered in ``param_values`` and can be swapped for callables or
    Forcing objects at any time.  Callables must follow the contract in
    ``contracts/signal_model_contract.md``.

    See also
    --------
    EBM0D : Zero-dimensional variant.
    EBMBase : Shared base class.
    albedo_func1D : Latitudinally-resolved albedo callable.

    Examples
    --------

    .. code-block:: python

        from paleobeasts.signal_models.ebm import EBM1DLat

        model = EBM1DLat(forcing=None, S0=1365.0, grid_n=50)
        output = model.integrate(
            t_span=(0, 200), y0=[15.0], method='rk4', dt=1.0
        )
        # output.diagnostic_variables['Tglobal'] gives time series of global mean T

    With a CO2 ramp:

    .. code-block:: python

        import paleobeasts as pb

        co2_ramp = pb.core.Forcing.from_sequence([
            pb.core.Hold(duration=100, value=0.0),
            pb.core.Ramp(duration=100, y0=0.0, yf=4.0, shape='linear'),
        ])
        model = EBM1DLat(forcing=None, CO2_forcing=co2_ramp)
        output = model.integrate(t_span=(0, 200), y0=[15.0], method='rk4', dt=1.0)

    """

    uses_post_history = True

    def __init__(self, forcing=None, var_name='ebm1d_lat', grid_n=50, C=10.0, D=0.55,
                 A=210.0, B=2.0, S0=1365.0, CO2_forcing=0.0,
                 state_variables=None, diagnostic_variables=None):
        self.grid_n = int(grid_n)
        if self.grid_n < 3:
            raise ValueError("grid_n must be at least 3.")

        self.phi = np.linspace(-90.0, 90.0, self.grid_n)
        self.x = np.sin(np.deg2rad(self.phi))

        if state_variables is None:
            state_variables = [f'T_{i}' for i in range(self.grid_n)]
        if diagnostic_variables is None:
            diagnostic_variables = ['ice_line_lat', 'Tglobal']

        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables)

        self.C = C
        self.D = D
        self.A = A
        self.B = B
        self.S0 = S0
        self.CO2_forcing = CO2_forcing
        self._transport_scale = np.pi / 2.0
        self.param_values = {
            'C': C, 'D': D, 'A': A, 'B': B, 'S0': S0, 'CO2_forcing': CO2_forcing,
        }
        self.params = ()

    def validate_initial_state(self, y0):
        """Validate and normalize the initial temperature profile.

        Overrides :meth:`PBModel.validate_initial_state` to accept a scalar
        initial condition and broadcast it uniformly across the latitude grid.

        Parameters
        ----------
        y0 : float or array-like
            Initial temperature(s).  A scalar is broadcast to all ``grid_n``
            grid points.  An array must have length exactly ``grid_n``.

        Returns
        -------
        y0_arr : ndarray of float, shape (grid_n,)
            Validated and grid-length initial temperature array.
        """
        y0_arr = np.asarray(y0, dtype=float).reshape(-1)
        if y0_arr.size == 1:
            return np.full(self.grid_n, float(y0_arr[0]), dtype=float)
        if y0_arr.size != self.grid_n:
            raise ValueError(
                f"Initial state length {y0_arr.size} does not match grid_n ({self.grid_n})."
            )
        return y0_arr

    def calc_albedo(self, T, t):
        """Compute the latitudinal ice-albedo with a linear transition zone.

        Overrides :meth:`EBMBase.calc_albedo`.  Each grid point is assigned
        an albedo based on local temperature: 0.6 below -10 °C, 0.3 above
        0 °C, and a linear blend in between.

        Parameters
        ----------
        T : array-like, shape (grid_n,)
            Current temperature profile (°C or K — the threshold values
            -10 and 0 assume °C; ensure consistency with initial conditions).
        t : float
            Current time.  Unused; kept for a uniform external signature.

        Returns
        -------
        albedo : ndarray of float, shape (grid_n,)
            Albedo at each latitude grid point, in [0.3, 0.6].
        """
        temperature = np.asarray(T, dtype=float)
        albedo = np.empty_like(temperature)
        cold = temperature < -10.0
        warm = temperature > 0.0
        transition = (~cold) & (~warm)
        albedo[cold] = 0.6
        albedo[warm] = 0.3
        albedo[transition] = 0.6 - 0.3 * ((temperature[transition] + 10.0) / 10.0)
        return albedo

    def calc_OLR(self, T, t):
        """Compute the Budyko linear OLR: ``(A - CO2_forcing) + B * T``.

        Overrides :meth:`EBMBase.calc_OLR`.  Parameters ``A``, ``B``, and
        ``CO2_forcing`` are resolved through ``get_param_value``, so they can
        be time-varying or Forcing objects.

        Parameters
        ----------
        T : array-like, shape (grid_n,)
            Current temperature profile.
        t : float
            Current time, passed to ``get_param_value`` for time-varying params.

        Returns
        -------
        olr : ndarray of float, shape (grid_n,)
            Outgoing longwave radiation at each latitude grid point (W m⁻²).
        """
        A = self.get_param_value('A', t, T)
        CO2_forcing = self.get_param_value('CO2_forcing', t, T)
        B = self.get_param_value('B', t, T)
        return (A - CO2_forcing) + B * T

    def annual_mean_insolation(self, t, state):
        """Compute annual-mean insolation with a P2 latitudinal distribution.

        Approximates the zonal-mean annual-mean absorbed solar radiation as:

            S(x) = (S0 / 4) * s(x),    s(x) = 1 - 0.482 * P2(sin phi)

        where P2 is the second Legendre polynomial and x = sin(phi).

        Parameters
        ----------
        t : float
            Current time, passed to ``get_param_value`` for time-varying S0.
        state : array-like
            Current state vector (used only to satisfy the callable contract
            when resolving ``S0``).

        Returns
        -------
        insolation : ndarray of float, shape (grid_n,)
            Annual-mean insolation at each latitude grid point (W m⁻²).
        """
        sin_phi = self.x
        s = 1.0 - 0.482 * (3.0 * sin_phi ** 2 - 1.0) / 2.0
        S0 = self.get_param_value('S0', t, state)
        return 0.25 * S0 * s

    def calc_diffusion(self, temperature, t, state):
        """Compute meridional heat diffusion in x = sin(phi) coordinates.

        Applies no-flux boundary conditions at the poles by setting the
        diffusive flux to zero at the first and last grid points before
        taking the divergence.

        Parameters
        ----------
        temperature : array-like, shape (grid_n,)
            Current temperature profile.
        t : float
            Current time, passed to ``get_param_value`` for time-varying D.
        state : array-like
            Current full state vector (used only to satisfy the callable
            contract when resolving ``D``).

        Returns
        -------
        diffusion : ndarray of float, shape (grid_n,)
            Diffusive heat flux convergence at each grid point (W m⁻²),
            scaled by the transport factor ``pi/2``.
        """
        D = self.get_param_value('D', t, state) * self._transport_scale
        x = self.x
        dTdx = np.gradient(temperature, x, edge_order=2)
        flux = (1.0 - x ** 2) * dTdx
        flux[0] = 0.0
        flux[-1] = 0.0
        return D * np.gradient(flux, x, edge_order=2)

    def calc_global_mean(self, temperature):
        """Compute the cosine-weighted global mean temperature.

        Parameters
        ----------
        temperature : array-like, shape (grid_n,)
            Zonal-mean temperature profile at one timestep.

        Returns
        -------
        Tglobal : float
            Area-weighted global mean temperature in the same units as input.
        """
        weights = np.cos(np.deg2rad(self.phi))
        return float(np.average(np.asarray(temperature, dtype=float), weights=weights))

    def calc_ice_line_lat(self, temperature):
        """Interpolate the ice-line latitude from the temperature profile.

        Defines the ice line as the -10 °C isotherm.  Computes the ice-line
        latitude independently in each hemisphere using linear interpolation
        between the two adjacent grid points that straddle the threshold, then
        returns the average of the two hemispheric values.

        Special cases: returns 90° if a hemisphere is entirely above the
        threshold (no ice) and 0° if entirely at or below it (full ice cover).

        Parameters
        ----------
        temperature : array-like, shape (grid_n,)
            Zonal-mean temperature profile at one timestep.

        Returns
        -------
        ice_line_lat : float
            Mean hemispheric ice-line latitude in degrees from the equator
            (range [0°, 90°]).
        """
        temperature = np.asarray(temperature, dtype=float)
        threshold = -10.0
        abs_phi = np.abs(self.phi)
        hemi_edges = []

        for mask in (self.phi >= 0.0, self.phi <= 0.0):
            phi_side = abs_phi[mask]
            temp_side = temperature[mask]
            order = np.argsort(phi_side)
            phi_side = phi_side[order]
            temp_side = temp_side[order]

            if np.all(temp_side > threshold):
                hemi_edges.append(90.0)
                continue
            if np.all(temp_side <= threshold):
                hemi_edges.append(0.0)
                continue

            cold_idx = np.where(temp_side <= threshold)[0][0]
            warm_idx = cold_idx - 1
            t_warm = temp_side[warm_idx]
            t_cold = temp_side[cold_idx]
            phi_warm = phi_side[warm_idx]
            phi_cold = phi_side[cold_idx]
            frac = (threshold - t_warm) / (t_cold - t_warm)
            hemi_edges.append(float(phi_warm + frac * (phi_cold - phi_warm)))

        return float(np.mean(hemi_edges))

    def dydt(self, t, state):
        """Evaluate the right-hand side of the PDE at time t and state.

        Computes the net energy flux at each latitude grid point from
        insolation, ice-albedo feedback, Budyko OLR, and meridional diffusion.

        This method has **no side effects**: because ``uses_post_history = True``,
        all output is derived from the full solved trajectory in
        :meth:`populate_diagnostics_from_history` rather than accumulated here.

        Parameters
        ----------
        t : float
            Current time.
        state : array-like, shape (grid_n,)
            Current zonal-mean temperature profile.

        Returns
        -------
        dTdt : ndarray of float, shape (grid_n,)
            Time-derivative of the temperature at each latitude grid point.
        """
        temperature = np.asarray(state, dtype=float)
        C = self.calc_C(temperature, t)
        insolation = self.annual_mean_insolation(t, state)
        albedo = self.calc_albedo(temperature, t)
        absorbed_sw = insolation * (1.0 - albedo)
        olr = self.calc_OLR(temperature, t)
        diffusion = self.calc_diffusion(temperature, t, state)
        return (absorbed_sw - olr + diffusion) / C

    def populate_diagnostics_from_history(self, time, history):
        """Compute diagnostic variables from the full solved trajectory.

        Called automatically by :meth:`PBModel.post_integrate` after the
        solver completes.  Populates ``self.diagnostic_variables`` with the
        global-mean temperature and ice-line latitude at every timestep.

        Parameters
        ----------
        time : array-like, shape (n_steps,)
            Solver time axis.
        history : ndarray, shape (n_steps, grid_n)
            Full temperature trajectory; each row is the temperature profile
            at one timestep.
        """
        self.diagnostic_variables['Tglobal'] = np.array(
            [self.calc_global_mean(row) for row in history], dtype=float,
        )
        self.diagnostic_variables['ice_line_lat'] = np.array(
            [self.calc_ice_line_lat(row) for row in history], dtype=float,
        )
