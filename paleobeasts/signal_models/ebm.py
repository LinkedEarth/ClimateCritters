import numpy as np
from functools import partial
from ..utils import constants as phys
from ..core.pbmodel import PBModel


class EBM(PBModel):
    """Energy Balance Model (EBM) class

    A simple model that describes the evolution of the temperature of the Earth's surface
    as a function of the radiative balance between incoming solar radiation and outgoing longwave radiation. The model
    is described by the following ordinary differential equation:

    $C \frac{dT}{dt} = (1 - \alpha) \frac{S_0}{4} - OLR$

    where $C$ is the heat capacity of the Earth's surface, $\alpha$ is the albedo, $S_0$ is the incoming solar radiation,
    $OLR$ is the outgoing longwave radiation.

    #TODO: implement meridional heat transport; currently limited to 0D


    Parameters
    ----------
    forcing : pb.Forcing
        Object that provides the forcing at time t.

    state_variables : list
        List of state variables for the model. Default is ['T'].

    diagnostic_variables : list
        List of diagnostic variables for the model. Default is ['albedo', 'absorbed_SW', 'OLR', 'solar_incoming'].

    var_name : str
        Name of the variable being modeled. Default is 'temperature'.

    OLR : float or function or pb.Forcing
        Function that calculates the outgoing longwave radiation as a function of temperature. Default is the Stefan-Boltzmann law with prad=650, ps=1000.
        To specify different values, pass OLR_func(pRad, ps) where pRad is the radiative pressure and ps is the surface pressure.

    C : float or callable or pb.Forcing
        Heat capacity of the Earth's surface. Default is 4.

    albedo : float or function or pb.Forcing
        Albedo of the Earth's surface. Default is 0.3.

    Notes
    -----
    Parameters may be constants, callables, or ``pb.Forcing`` objects. Callables can use
    common signatures such as ``(t)``, ``(t, state)``, ``(t, state, model)``,
    ``(model, state)``, or ``(state)``.

    """
    def __init__(self, forcing, state_variables= ['T'],
                 diagnostic_variables = ['albedo', 'absorbed_SW', 'OLR', 'solar_incoming'],
                 var_name='temperature', OLR=None, C=4, merid_diff=0, albedo=.3):
        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables )

        self.forcing = forcing
        self.variable_name = var_name
        self.C = C
        self.params = ()
        self.albedo = albedo
        self.OLR = OLR if OLR is not None else OLR_func()
        self.merid_diff = merid_diff
        self.phi = np.linspace(-np.pi / 2, np.pi / 2, 100)
        self.param_values = {
            'C': self.C,
            'albedo': self.albedo,
            'OLR': self.OLR,
            'merid_diff': self.merid_diff,
        }

    def dydt(self, t, x):

        T =x
        if isinstance(T, np.ndarray):
            T = T[-1]

        # assumes forcing is S0
        # 1/4 factor because Earth emits radiation over full surface (4πR2)
        # but at any given time only receives incoming (solar) radiation over its cross-sectional area, πR2
        f_solar_incoming = self.forcing.get_forcing(t)
        albedo = self.calc_albedo(T, t)
        absorbed_SW = (1 - albedo) * f_solar_incoming/4
        OLR = self.calc_OLR(T, t)

        C = self.calc_C(T, t)
        dTdt = 1 / C * (absorbed_SW - OLR + self.calc_merid_diff(T, t))

        new_row = np.array([(T)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        self.diagnostic_variables['albedo'].append(albedo)
        self.diagnostic_variables['absorbed_SW'].append(absorbed_SW)
        self.diagnostic_variables['OLR'].append(OLR)
        self.diagnostic_variables['solar_incoming'].append(f_solar_incoming)
        
        if t>0:
            self.time.append(t)

        return [dTdt]

    def calc_OLR(self, T, t):
        return self.get_param_value('OLR', t, T)

    def calc_albedo(self, T, t):
        return self.get_param_value('albedo', t, T)

    def calc_merid_diff(self, T, t):
        return self.get_param_value('merid_diff', t, T)

    def calc_C(self, T, t):
        return self.get_param_value('C', t, T)



def albedo_func(ebm_model, Ts, alpha_ice=.6, alpha_0=.3, T1=260., T2=290.):
    if isinstance(Ts, np.ndarray):
        Ts = Ts[-1]
    if Ts < T1:
        a0 = alpha_ice
    elif (Ts >= T1) & (Ts <= T2):
        r = (Ts - T2) ** 2 / (T2 - T1) ** 2
        a0 = alpha_0 + (alpha_ice - alpha_0) * r
    else:
        a0 = alpha_0

    return a0


def albedo_func1D(ebm_model, Ts, a2=.25, alpha_ice=.6, alpha_0=.1, T1=260., T2=290.):
    phi = ebm_model.phi
    a0 = albedo_func(Ts, alpha_ice, alpha_0, T1, T2)
    P2 = .5 * (3 * np.sin(phi) ** 2 - 1)
    return a0 + a2 * P2


def _OLR_func(Ts, pRad=650, ps=1000):
    if isinstance(Ts, np.ndarray):
        Ts = Ts[-1]
        # print(Ts)
    Te = (pRad / ps) ** (2. / 7.) * Ts  # emission temperature assuming dry adiabat
    # print('Te', Te)
    return phys.sigma * (Te ** 4.)

def OLR_func(pRad=650, ps=1000):
    return partial(_OLR_func, pRad=pRad, ps=ps)
    # if isinstance(Ts, np.ndarray):
    #     Ts = Ts[-1]
    #     # print(Ts)
    # Te = (pRad / ps) ** (2. / 7.) * Ts  # emission temperature assuming dry adiabat
    # # print('Te', Te)
    # return phys.sigma * (Te ** 4.)


### Future support for 1D models
def incoming_SW_func(t, S_0=1360.8):
    return np.ones(np.array(t).shape)*S_0

# def advection_diffusion(ebm_model, T, D):
#     phi = ebm_model.phi
#     return D / np.cos(phi) * (np.gradient(phi, T) * -np.sin(phi) + np.gradient(T, phi) * np.cos(phi))

def calc_f(t, S_0=1360.8, T1=11):
    return S_0+ np.sin(t*(2 * np.pi)/T1)
