import numpy as np

from paleobeasts.core.pbmodel import PBModel


class LatitudinalEBM(PBModel):
    """Diffusive annual-mean latitudinal energy balance model."""

    def __init__(
        self,
        forcing=None,
        var_name='latitudinal_ebm',
        grid_n=50,
        C=10.0,
        D=0.55,
        A=210.0,
        B=2.0,
        S0=1365.0,
        CO2_forcing=0.0,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        self.grid_n = int(grid_n)
        if self.grid_n < 3:
            raise ValueError("grid_n must be at least 3.")

        self.phi = np.linspace(-90.0, 90.0, self.grid_n)
        self.x = np.sin(np.deg2rad(self.phi))

        if state_variables is None:
            state_variables = [f'T_{i}' for i in range(self.grid_n)]
        if diagnostic_variables is None:
            diagnostic_variables = ['ice_line_lat', 'Tglobal']

        super().__init__(
            forcing,
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.C = C
        self.D = D
        self.A = A
        self.B = B
        self.S0 = S0
        self.CO2_forcing = CO2_forcing
        # The diffusion operator is written in x = sin(phi) while the model grid
        # is specified in latitude degrees. This factor keeps the meridional
        # transport strength in the expected annual-mean range across grid sizes.
        self._transport_scale = np.pi / 2.0
        self.param_values = {
            'C': C,
            'D': D,
            'A': A,
            'B': B,
            'S0': S0,
            'CO2_forcing': CO2_forcing,
        }
        self.params = ()

    uses_post_history = True

    def validate_initial_state(self, y0):
        y0_arr = np.asarray(y0, dtype=float).reshape(-1)
        if y0_arr.size == 1:
            return np.full(self.grid_n, float(y0_arr[0]), dtype=float)
        if y0_arr.size != self.grid_n:
            raise ValueError(
                f"Initial state length {y0_arr.size} does not match grid_n ({self.grid_n})."
            )
        return y0_arr

    def annual_mean_insolation(self, t, state):
        sin_phi = self.x
        s = 1.0 - 0.482 * (3.0 * sin_phi ** 2 - 1.0) / 2.0
        S0 = self.get_param_value('S0', t, state)
        return 0.25 * S0 * s

    def calc_albedo(self, temperature):
        temperature = np.asarray(temperature, dtype=float)
        albedo = np.empty_like(temperature)
        cold = temperature < -10.0
        warm = temperature > 0.0
        transition = (~cold) & (~warm)

        albedo[cold] = 0.6
        albedo[warm] = 0.3
        albedo[transition] = 0.6 - 0.3 * ((temperature[transition] + 10.0) / 10.0)
        return albedo

    def calc_diffusion(self, temperature, t, state):
        D = self.get_param_value('D', t, state) * self._transport_scale
        x = self.x
        dTdx = np.gradient(temperature, x, edge_order=2)
        flux = (1.0 - x ** 2) * dTdx
        flux[0] = 0.0
        flux[-1] = 0.0
        return D * np.gradient(flux, x, edge_order=2)

    def calc_global_mean(self, temperature):
        weights = np.cos(np.deg2rad(self.phi))
        return float(np.average(np.asarray(temperature, dtype=float), weights=weights))

    def calc_ice_line_lat(self, temperature):
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
        C = self.get_param_value('C', t, state)
        A = self.get_param_value('A', t, state)
        B = self.get_param_value('B', t, state)
        CO2_forcing = self.get_param_value('CO2_forcing', t, state)

        insolation = self.annual_mean_insolation(t, state)
        albedo = self.calc_albedo(temperature)
        absorbed_sw = insolation * (1.0 - albedo)
        olr = (A - CO2_forcing) + B * temperature
        diffusion = self.calc_diffusion(temperature, t, state)

        return (absorbed_sw - olr + diffusion) / C

    def populate_diagnostics_from_history(self, time, history):
        self.diagnostic_variables['Tglobal'] = np.array(
            [self.calc_global_mean(row) for row in history],
            dtype=float,
        )
        self.diagnostic_variables['ice_line_lat'] = np.array(
            [self.calc_ice_line_lat(row) for row in history],
            dtype=float,
        )
