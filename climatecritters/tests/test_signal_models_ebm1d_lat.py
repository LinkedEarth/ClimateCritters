''' Tests for climatecritters.model_critters.ebm.EBM1DLat

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import numpy as np

from climatecritters.model_critters import EBM1DLat


class TestSignalModelsEBM1DLat:
    def test_import_and_equilibrate_t0(self):
        model = EBM1DLat(S0=1365.0)
        output = model.integrate(t_span=(0, 200), y0=[15.0])

        assert len(output.diagnostic_variables['Tglobal']) == len(output.time)
        assert len(output.diagnostic_variables['ice_line_lat']) == len(output.time)
        assert np.isclose(output.diagnostic_variables['Tglobal'][-1], 15.0, atol=5.0)

    def test_cold_state_low_latitude_ice_t0(self):
        model = EBM1DLat(S0=1200.0)
        output = model.integrate(t_span=(0, 200), y0=[0.0])

        assert np.isfinite(output.diagnostic_variables['ice_line_lat'][-1])
        assert output.diagnostic_variables['ice_line_lat'][-1] < 40.0

    def test_grid_sizes_t0(self):
        for grid_n in (30, 100):
            model = EBM1DLat(grid_n=grid_n, S0=1365.0)
            output = model.integrate(t_span=(0, 50), y0=[15.0])

            assert len(output.state_variables.dtype.names) == grid_n
            assert np.isfinite(output.diagnostic_variables['Tglobal'][-1])
