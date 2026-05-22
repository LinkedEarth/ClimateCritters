''' Tests for paleobeasts.core.pbmodel time axis utilities

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}
'''

import numpy as np
import pytest
import paleobeasts as pb

from paleobeasts.core.pbmodel import PBModel
from paleobeasts.signal_models import lorenz


class TestCorePBModelReframeTimeAxis:
    def test_reframe_time_axis_rk45_t0(self):
        def func(x):
            return 0

        forcing = pb.core.Forcing(func)
        model = lorenz.Lorenz63(forcing=forcing)
        model.integrate(t_span=(0, 5), y0=[1, 1, 1], method='RK45')

        t_eval = np.linspace(0, 5, 51)
        reframed = model.reframe_time_axis(t_eval, update_state=False)

        assert len(reframed) == len(t_eval)
        assert set(reframed.dtype.names) == {'x', 'y', 'z'}

    def test_reframe_time_axis_euler_t0(self):
        def func(x):
            return 0

        forcing = pb.core.Forcing(func)
        model = lorenz.Lorenz63(forcing=forcing)
        model.integrate(t_span=(0, 5), y0=[1, 1, 1], method='euler', kwargs={'dt': 0.1})

        t_eval = np.linspace(0, 5, 26)
        reframed = model.reframe_time_axis(t_eval, update_state=True)

        assert len(reframed) == len(t_eval)
        assert np.allclose(model.time, t_eval)


class _PostHistoryModel(PBModel):
    def __init__(self):
        super().__init__(forcing=None, variable_name='post_history', state_variables=['x'],
                         diagnostic_variables=['x_squared'])

    uses_post_history = True

    def dydt(self, t, x):
        return [-x[0]]

    def populate_diagnostics_from_history(self, time, history):
        self.diagnostic_variables['x_squared'] = history[:, 0] ** 2


class TestCorePBModelPostHistoryHooks:
    def test_post_history_model_integrates_t0(self):
        model = _PostHistoryModel()
        model.integrate(t_span=(0, 1), y0=[1.0], method='euler', kwargs={'dt': 0.1})

        assert model.state_variables.dtype.names == ('x',)
        assert len(model.time) == len(model.diagnostic_variables['x_squared'])
        assert np.isclose(model.state_variables['x'][0], 1.0)


class _ParamContractModel(PBModel):
    def __init__(self, coeff=1.0):
        super().__init__(
            forcing=None,
            variable_name='param_contract',
            state_variables=['x'],
            diagnostic_variables=[],
        )
        self.coeff = coeff
        self.param_values = {'coeff': coeff}

    def dydt(self, t, x):
        coeff = self.get_param('coeff', t, x)
        return [coeff * x[0]]


class TestCorePBModelParameterContract:
    def test_callable_accepts_supported_signatures_t0(self):
        model_t = _ParamContractModel(coeff=lambda t: 2.0)
        model_ts = _ParamContractModel(coeff=lambda t, state: 2.0)
        model_tsm = _ParamContractModel(coeff=lambda t, state, model: 2.0)

        assert model_t.get_param('coeff', 0.0, [1.0]) == 2.0
        assert model_ts.get_param('coeff', 0.0, [1.0]) == 2.0
        assert model_tsm.get_param('coeff', 0.0, [1.0]) == 2.0

    def test_non_compliant_callable_raises_t0(self):
        model = _ParamContractModel(coeff=lambda model, state: 2.0)
        with pytest.raises(TypeError):
            model.get_param('coeff', 0.0, [1.0])

    def test_attribute_assignment_syncs_param_values_t0(self):
        model = _ParamContractModel(coeff=1.0)
        model.coeff = lambda t: 3.0
        assert model.param_values['coeff'](0.0) == 3.0


class _FunctionSwapModel(PBModel):
    def __init__(self):
        super().__init__(forcing=None, variable_name='function_swap', state_variables=['x'])

    def calc_term(self, value):
        return value + 1


class TestCorePBModelSetFunction:
    def test_set_function_plain_callable_t0(self):
        model = _FunctionSwapModel()

        def plain_calc_term(value):
            return value + 5

        model.set_function('calc_term', plain_calc_term)
        assert model.calc_term(2) == 7

    def test_set_function_bound_callable_t1(self):
        model = _FunctionSwapModel()

        def bound_calc_term(self, value):
            return value + 9

        model.set_function('calc_term', bound_calc_term)
        assert model.calc_term(1) == 10

    def test_set_function_force_bind_t2(self):
        model = _FunctionSwapModel()

        def bound_calc_term(self, value):
            return value + 4

        model.set_function('calc_term', bound_calc_term, bind=True)
        assert model.calc_term(3) == 7

    def test_set_function_errors_t3(self):
        model = _FunctionSwapModel()
        with pytest.raises(AttributeError):
            model.set_function('missing_function', lambda x: x)
        with pytest.raises(TypeError):
            model.set_function('calc_term', 123)
