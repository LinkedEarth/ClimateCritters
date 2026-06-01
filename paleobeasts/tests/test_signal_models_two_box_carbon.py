import numpy as np
import paleobeasts as pb

from paleobeasts.signal_models import BoxModelSpec, TwoBoxCarbon


class TestSignalModelsTwoBoxCarbon:
    def test_import_and_integrate_t0(self):
        model = TwoBoxCarbon(k=0.2, R=0.0, l_s=0.0)
        model.integrate(t_span=(0, 10), y0=[100.0, 50.0])

        assert model.state_variables.dtype.names == ("A", "S")
        assert "net_flux" in model.diagnostic_variables
        assert np.all(np.isfinite(model.diagnostic_variables["net_flux"]))

    def test_closed_system_conserves_mass_t0(self):
        model = TwoBoxCarbon(k=0.2, R=0.0, l_s=0.0, V_atm=1.0, V_surf=50.0)
        model.integrate(t_span=(0, 100), y0=[100.0, 200.0], method="RK45")

        total = model.state_variables["A"] + model.state_variables["S"]
        assert np.max(np.abs(total - total[0])) < 1e-8

    def test_forced_system_reaches_steady_state_t0(self):
        model = TwoBoxCarbon(k=0.2, R=1.0, l_s=0.05)
        model.integrate(t_span=(0, 100), y0=[0.0, 0.0], method="RK45")

        A_eq = 1.0 / 0.05
        S_eq = model.V_surf * (A_eq / model.V_atm)
        assert np.isclose(model.state_variables["A"][-1], A_eq, atol=2.0)
        assert np.isclose(model.state_variables["S"][-1], S_eq, atol=2.5)


class TestSignalModelsGenericBoxModel:
    def test_make_two_box_carbon_from_spec_t0(self):
        spec = BoxModelSpec("two_box_spec")
        spec.register_state_variables(["A", "S"])
        spec.register_diagnostic_variables(["net_flux"])
        spec.register_parameters(k=0.2, R=0.0, l_s=0.0, V_atm=1.0, V_surf=1.0)
        spec.register_input("R", fallback_param="R")

        def exchange_flux(ctx):
            return ctx.param("k") * (
                ctx["A"] / ctx.param("V_atm") - ctx["S"] / ctx.param("V_surf")
            )

        spec.register_relations({
            "A": lambda ctx: -exchange_flux(ctx) + ctx.input("R") - ctx.param("l_s") * ctx["A"],
            "S": lambda ctx: exchange_flux(ctx),
        })
        spec.register_diagnostic(
            "net_flux",
            lambda ctx: -exchange_flux(ctx) + ctx.input("R") - ctx.param("l_s") * ctx["A"],
        )

        model = spec.make_boxmodel()
        model.integrate(t_span=(0, 50), y0=[100.0, 50.0], method="RK45")

        assert model.state_variables.dtype.names == ("A", "S")
        assert "net_flux" in model.diagnostic_variables
        assert np.all(np.isfinite(model.diagnostic_variables["net_flux"]))

    def test_make_stocker_style_box_model_t0(self):
        spec = BoxModelSpec("stocker_box")
        spec.register_statevariables(["Ts"])
        spec.register_diagnosticvariables(["Tn"])
        spec.register_parameters(tau=1000.0, beta=-1.0, Tn=0.0)
        spec.register_input("Tn", fallback_param="Tn")
        spec.register_relations({
            "Ts": lambda ctx: (ctx.param("beta") * ctx.input("Tn") - ctx["Ts"]) / ctx.param("tau"),
        })

        north = pb.Forcing(lambda t: 1.0)
        model = spec.make_boxmodel()
        model.register_forcing('Tn', north)
        model.integrate(t_span=(0, 100), y0=[0.0], method="RK45")

        assert model.state_variables.dtype.names == ("Ts",)
        assert np.all(np.isfinite(model.diagnostic_variables["Tn"]))

    def test_automatic_exchange_network_conserves_mass_t0(self):
        spec = BoxModelSpec("exchange_network")
        spec.register_state_variables(["A", "S"])
        spec.register_box_volumes(A=1.0, S=50.0)
        spec.register_exchange("A", "S", 0.2)

        model = spec.make_boxmodel()
        model.integrate(t_span=(0, 100), y0=[100.0, 200.0], method="RK45")

        total = model.state_variables["A"] + model.state_variables["S"]
        assert np.max(np.abs(total - total[0])) < 1e-8

    def test_automatic_transport_network_moves_mass_directionally_t0(self):
        spec = BoxModelSpec("transport_network")
        spec.register_state_variables(["upstream", "downstream"])
        spec.register_box_volumes(upstream=1.0, downstream=1.0)
        spec.register_transport("upstream", "downstream", 0.3)
        spec.register_diagnostic("total_mass", lambda ctx: ctx["upstream"] + ctx["downstream"])

        model = spec.make_boxmodel()
        model.integrate(t_span=(0, 20), y0=[10.0, 0.0], method="RK45")

        assert model.state_variables["upstream"][-1] < model.state_variables["upstream"][0]
        assert model.state_variables["downstream"][-1] > model.state_variables["downstream"][0]
        assert np.max(np.abs(model.diagnostic_variables["total_mass"] - 10.0)) < 1e-8
