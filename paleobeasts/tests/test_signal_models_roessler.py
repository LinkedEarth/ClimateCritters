import numpy as np
import paleobeasts as pb

from paleobeasts.signal_models import Roessler


class TestSignalModelsRoessler:
    def test_import_and_integrate_t0(self):
        model = Roessler(forcing=None, a=0.2, b=0.2, c=5.7)
        model.integrate(t_span=(0, 100), y0=[1, 1, 1])

        assert model.state_variables.dtype.names == ('x', 'y', 'z')
        assert len(model.time) == len(model.state_variables)

    def test_bounded_attractor_t0(self):
        model = Roessler(forcing=None, a=0.2, b=0.2, c=5.7)
        model.integrate(t_span=(0, 200), y0=[1, 1, 1])

        assert np.max(np.abs(model.state_variables['x'])) < 30

    def test_time_varying_params_via_forcing_t0(self):
        const_a = pb.core.Forcing(lambda t: 0.2)
        const_b = pb.core.Forcing(lambda t: 0.2)
        const_c = pb.core.Forcing(lambda t: 5.7)
        model = Roessler(forcing=None, a=const_a, b=const_b, c=const_c)

        model.integrate(t_span=(0, 10), y0=[1, 1, 1], method='euler', dt=0.01)

        assert model.state_variables.dtype.names == ('x', 'y', 'z')
