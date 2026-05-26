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
    """Stefan-Boltzmann OLR assuming a dry-adiabatic lapse rate.

    Compliant callable: (t, state).
    """
    Ts = float(np.asarray(state).reshape(-1)[0])
    Te = (pRad / ps) ** (2. / 7.) * Ts
    return phys.sigma * Te ** 4.


def OLR_func(pRad=650, ps=1000):
    """Return a compliant (t, state) callable for Stefan-Boltzmann OLR.

    Parameters
    ----------
    pRad : float
        Radiative pressure level (hPa).
    ps : float
        Surface pressure (hPa).
    """
    def _olr(t, state):
        return _olr_stefan_boltzmann(t, state, pRad, ps)
    return _olr


def albedo_func(t, state, alpha_ice=0.6, alpha_0=0.3, T1=260., T2=290.):
    """Temperature-dependent albedo with smooth ice-line transition.

    Compliant callable: (t, state).

    Parameters
    ----------
    t : float
        Current time (unused; required for contract compliance).
    state : float or array-like
        Current temperature. The first element is used if array-like.
    alpha_ice : float
        Albedo for cold (ice-covered) state.
    alpha_0 : float
        Albedo for warm (ice-free) state.
    T1 : float
        Temperature below which full ice albedo applies.
    T2 : float
        Temperature above which full warm albedo applies.
    """
    Ts = float(np.asarray(state).reshape(-1)[0])
    return _scalar_albedo(Ts, alpha_ice, alpha_0, T1, T2)


def albedo_func1D(t, state, model, a2=0.25, alpha_ice=0.6, alpha_0=0.1, T1=260., T2=290.):
    """P2-corrected latitudinal albedo (Legendre polynomial parameterization).

    Compliant callable: (t, state, model).

    Uses the global-mean temperature to set the base albedo, then adds a
    second Legendre polynomial correction to capture the equator-to-pole
    gradient. Requires the model to expose a ``phi`` attribute (degrees).

    Parameters
    ----------
    t : float
        Current time (unused; required for contract compliance).
    state : array-like
        Full temperature array (grid-length). Global mean is taken internally.
    model : EBMBase subclass
        Model instance; must have a ``phi`` attribute (latitude in degrees).
    a2 : float
        Amplitude of the P2 Legendre polynomial correction.
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

    Subclasses with different OLR or albedo formulations (e.g. ``EBM1DLat``
    with Budyko linear OLR and an array ice-line albedo) override the relevant
    ``calc_*`` methods. The external signature ``(self, T, t)`` is shared across
    all variants so that user code can call these helpers uniformly.
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
    """Zero-dimensional energy balance model.

    Evolves global-mean surface temperature T according to:

        C dT/dt = (1 - alpha) * S0/4 - OLR

    where S0 is supplied by ``forcing``, ``alpha`` is the planetary albedo,
    and OLR is the outgoing longwave radiation.

    Parameters
    ----------
    forcing : pb.Forcing
        Provides solar constant S0 at time t.
    state_variables : list of str, optional
        Default ``['T']``.
    diagnostic_variables : list of str, optional
        Default ``['albedo', 'absorbed_SW', 'OLR', 'solar_incoming']``.
    var_name : str
        Label for the modeled quantity. Default ``'temperature'``.
    OLR : callable or None
        Outgoing longwave radiation. Must have a compliant signature:
        ``(t)``, ``(t, state)``, or ``(t, state, model)``.
        Default is Stefan-Boltzmann via ``OLR_func(pRad=650, ps=1000)``.
    C : float or callable or pb.Forcing
        Heat capacity. Default 4.
    albedo : float or callable or pb.Forcing
        Planetary albedo. Default 0.3.

    Notes
    -----
    Parameters may be constants, callables, or ``pb.Forcing`` objects.
    Callables must follow the contract documented in
    ``contracts/signal_model_contract.md``.
    """

    def __init__(self, forcing, state_variables=None, diagnostic_variables=None,
                 var_name='temperature', OLR=None, C=4, albedo=0.3):
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
    """Diffusive annual-mean latitudinal energy balance model.

    Budyko-Sellers type 1D EBM on a latitude grid:

        C dT/dt = S(x)(1 - alpha(T)) - OLR(T) + D * div(grad T)

    where ``x = sin(phi)``, OLR is the Budyko linear form ``(A - CO2_forcing) + B*T``,
    and diffusion is computed in x-coordinates with no-flux polar boundaries.

    Overrides ``calc_OLR`` and ``calc_albedo`` from ``EBMBase`` with
    grid-aware array implementations.

    Parameters
    ----------
    forcing : pb.Forcing or None
        Optional external forcing (reserved for future use).
    var_name : str
        Label for the modeled quantity. Default ``'ebm1d_lat'``.
    grid_n : int
        Number of latitude grid points (≥ 3). Default 50.
    C : float or callable or pb.Forcing
        Heat capacity (W yr m⁻² K⁻¹).
    D : float or callable or pb.Forcing
        Meridional diffusion coefficient.
    A : float or callable or pb.Forcing
        Budyko OLR intercept (W m⁻²).
    B : float or callable or pb.Forcing
        Budyko OLR slope (W m⁻² K⁻¹).
    S0 : float or callable or pb.Forcing
        Solar constant (W m⁻²).
    CO2_forcing : float or callable or pb.Forcing
        Radiative forcing from CO2; shifts OLR intercept down.

    Notes
    -----
    ``uses_post_history = True``: diagnostics are populated from the full
    solved trajectory via ``populate_diagnostics_from_history``.
    ``validate_initial_state`` accepts a scalar and broadcasts it to the grid.
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
        """Accept a scalar (broadcast to full grid) or a grid-length array."""
        y0_arr = np.asarray(y0, dtype=float).reshape(-1)
        if y0_arr.size == 1:
            return np.full(self.grid_n, float(y0_arr[0]), dtype=float)
        if y0_arr.size != self.grid_n:
            raise ValueError(
                f"Initial state length {y0_arr.size} does not match grid_n ({self.grid_n})."
            )
        return y0_arr

    def calc_albedo(self, T, t):
        """Array ice-albedo: step function with linear -10 °C to 0 °C transition.

        Overrides ``EBMBase.calc_albedo``. The ``t`` argument is unused but
        kept for a consistent external signature.
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
        """Budyko linear OLR: ``(A - CO2_forcing) + B * T``.

        Overrides ``EBMBase.calc_OLR``. Parameters are resolved through
        ``get_param_value`` so they can be time-varying or Forcing objects.
        """
        A = self.get_param_value('A', t, T)
        CO2_forcing = self.get_param_value('CO2_forcing', t, T)
        B = self.get_param_value('B', t, T)
        return (A - CO2_forcing) + B * T

    def annual_mean_insolation(self, t, state):
        """Annual-mean insolation with P2 latitudinal distribution."""
        sin_phi = self.x
        s = 1.0 - 0.482 * (3.0 * sin_phi ** 2 - 1.0) / 2.0
        S0 = self.get_param_value('S0', t, state)
        return 0.25 * S0 * s

    def calc_diffusion(self, temperature, t, state):
        """Meridional diffusion in x = sin(phi) coordinates, no-flux at poles."""
        D = self.get_param_value('D', t, state) * self._transport_scale
        x = self.x
        dTdx = np.gradient(temperature, x, edge_order=2)
        flux = (1.0 - x ** 2) * dTdx
        flux[0] = 0.0
        flux[-1] = 0.0
        return D * np.gradient(flux, x, edge_order=2)

    def calc_global_mean(self, temperature):
        """Cosine-weighted global mean temperature."""
        weights = np.cos(np.deg2rad(self.phi))
        return float(np.average(np.asarray(temperature, dtype=float), weights=weights))

    def calc_ice_line_lat(self, temperature):
        """Interpolated ice-line latitude (average of NH and SH edges at -10 °C)."""
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
        temperature = np.asarray(state, dtype=float)
        C = self.calc_C(temperature, t)
        insolation = self.annual_mean_insolation(t, state)
        albedo = self.calc_albedo(temperature, t)
        absorbed_sw = insolation * (1.0 - albedo)
        olr = self.calc_OLR(temperature, t)
        diffusion = self.calc_diffusion(temperature, t, state)
        return (absorbed_sw - olr + diffusion) / C

    def populate_diagnostics_from_history(self, time, history):
        self.diagnostic_variables['Tglobal'] = np.array(
            [self.calc_global_mean(row) for row in history], dtype=float,
        )
        self.diagnostic_variables['ice_line_lat'] = np.array(
            [self.calc_ice_line_lat(row) for row in history], dtype=float,
        )
