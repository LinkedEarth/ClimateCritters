import numpy as np

from paleobeasts.signal_models import LatitudinalEBM


class TestSignalModelsLatitudinalEBM:
    def test_import_and_equilibrate_t0(self):
        model = LatitudinalEBM(forcing=None, S0=1365.0)
        model.integrate(t_span=(0, 200), y0=[15.0])

        assert len(model.diagnostic_variables['Tglobal']) == len(model.time)
        assert len(model.diagnostic_variables['ice_line_lat']) == len(model.time)
        assert np.isclose(model.diagnostic_variables['Tglobal'][-1], 15.0, atol=5.0)

    def test_cold_state_low_latitude_ice_t0(self):
        model = LatitudinalEBM(forcing=None, S0=1200.0)
        model.integrate(t_span=(0, 200), y0=[0.0])

        assert np.isfinite(model.diagnostic_variables['ice_line_lat'][-1])
        assert model.diagnostic_variables['ice_line_lat'][-1] < 40.0

    def test_grid_sizes_t0(self):
        for grid_n in (30, 100):
            model = LatitudinalEBM(forcing=None, grid_n=grid_n, S0=1365.0)
            model.integrate(t_span=(0, 50), y0=[15.0])

            assert len(model.state_variables.dtype.names) == grid_n
            assert np.isfinite(model.diagnostic_variables['Tglobal'][-1])
