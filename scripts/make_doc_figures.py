"""Generate static figures for the PaleoBeasts Reference documentation.

Run from the project root after a ``quartodoc build`` to populate
``docs/reference/figures/``.  The Quarto build does not require this script;
missing figures produce a broken-image placeholder but do not prevent the
site from building.

Usage
-----
Generate all figures::

    python scripts/make_doc_figures.py

Generate a single model's figure::

    python scripts/make_doc_figures.py --model Lorenz63

List available model names::

    python scripts/make_doc_figures.py --list
"""
# Agg backend must be set before any other matplotlib import
import matplotlib
matplotlib.use('Agg')

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIGURE_DIR = ROOT / 'docs' / 'reference' / 'figures'


def _save(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f'{name}_example.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f'  saved {path.relative_to(ROOT)}')


# ---------------------------------------------------------------------------
# EBMs
# ---------------------------------------------------------------------------

def fig_EBM0D():
    import paleobeasts as pb
    from paleobeasts.signal_models.ebm import EBM0D

    forcing = pb.core.Forcing(lambda t: 1360.0)
    model = EBM0D(forcing=forcing)
    output = model.integrate(t_span=(0, 500), y0=[288.0], method='RK45')
    ts = output.to_pyleo(var_names=['T'])
    fig, ax = ts.plot(label='T (K)')
    ax.set_title('EBM0D — global-mean temperature')
    _save('EBM0D')


def fig_EBM1DLat():
    import numpy as np
    from paleobeasts.signal_models.ebm import EBM1DLat

    grid_n = 50
    model = EBM1DLat(forcing=None, S0=1365.0, grid_n=grid_n)
    y0 = np.full(grid_n, 15.0)
    output = model.integrate(t_span=(0, 200), y0=y0, method='rk4', dt=1.0)

    T_final = np.array([output.state_variables[f'T_{i}'][-1] for i in range(grid_n)])
    phi = np.linspace(-90, 90, grid_n)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi, T_final, color='steelblue')
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('EBM1DLat — equilibrium temperature profile')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    fig.tight_layout()
    _save('EBM1DLat')


# ---------------------------------------------------------------------------
# Lorenz & Roessler
# ---------------------------------------------------------------------------

def fig_Lorenz63():
    import paleobeasts as pb
    from paleobeasts.signal_models.lorenz import Lorenz63

    model = Lorenz63(forcing=pb.core.Forcing(lambda t: 0.0))
    output = model.integrate(t_span=(0, 100), y0=[-8.0, 8.0, 27.0], method='RK45')

    x = output.state_variables['x']
    z = output.state_variables['z']

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, z, lw=0.3, alpha=0.8, color='steelblue', rasterized=True)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title('Lorenz 63 — phase portrait')
    fig.tight_layout()
    _save('Lorenz63')


def fig_Lorenz96():
    import numpy as np
    import paleobeasts as pb
    from paleobeasts.signal_models.lorenz import Lorenz96

    rng = np.random.default_rng(42)
    model = Lorenz96(forcing=None, n=40, F=8.0)
    y0 = rng.standard_normal(40) + 8.0
    output = model.integrate(t_span=(0, 10), y0=y0, method='rk4', dt=0.01)

    fig, ax = plt.subplots(figsize=(8, 3))
    for k in range(5):
        ts = output.state_variables[f'x{k}']
        ax.plot(output.time, ts, lw=0.8, label=f'x{k}')
    ax.set_xlabel('Time')
    ax.set_ylabel('X')
    ax.set_title('Lorenz 96 — first five slow variables')
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    _save('Lorenz96')


def fig_Roessler():
    import paleobeasts as pb
    from paleobeasts.signal_models.roessler import Roessler

    model = Roessler(forcing=None)
    output = model.integrate(t_span=(0, 200), y0=[0.1, 0.0, 0.0], method='RK45')

    x = output.state_variables['x']
    z = output.state_variables['z']

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, z, lw=0.3, alpha=0.8, color='darkorange', rasterized=True)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title('Rössler — phase portrait')
    fig.tight_layout()
    _save('Roessler')


# ---------------------------------------------------------------------------
# Pendulums & Oscillators
# ---------------------------------------------------------------------------

def fig_SimplePendulum():
    from paleobeasts.signal_models.pendulum import SimplePendulum

    model = SimplePendulum(forcing=None, L=1.0, g=9.81, damping=0.1)
    output = model.integrate(t_span=(0, 20), y0=[1.5, 0.0], method='RK45')
    ts = output.to_pyleo(var_names=['theta'])
    fig, ax = ts.plot(label='θ (rad)')
    ax.set_title('Simple pendulum — angle')
    _save('SimplePendulum')


def fig_DrivenPendulum():
    from paleobeasts.signal_models.pendulum import DrivenPendulum

    model = DrivenPendulum(forcing=None, q=0.5, A=1.2, Omega=2.0 / 3.0)
    output = model.integrate(
        t_span=(0, 500), y0=[0.0, 0.0], method='RK45',
        kwargs={'rtol': 1e-9, 'atol': 1e-11},
    )
    theta = output.state_variables['theta']
    omega = output.state_variables['omega']
    # wrap angle to [-pi, pi] for a cleaner portrait
    theta_w = (theta + np.pi) % (2 * np.pi) - np.pi

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(theta_w, omega, ',', ms=0.4, alpha=0.5, color='firebrick', rasterized=True)
    ax.set_xlabel('θ (rad, wrapped)')
    ax.set_ylabel('ω (rad/time)')
    ax.set_title('Driven pendulum — phase portrait')
    fig.tight_layout()
    _save('DrivenPendulum')


def fig_DoublePendulum():
    from paleobeasts.signal_models.pendulum import DoublePendulum

    model = DoublePendulum(forcing=None, m1=1.0, m2=1.0, L1=1.0, L2=1.0)
    output = model.integrate(
        t_span=(0, 60), y0=[np.pi / 2, 0.0, np.pi / 4, 0.0], method='RK45',
        kwargs={'rtol': 1e-10, 'atol': 1e-12},
    )
    t = output.time
    th1 = output.state_variables['theta1']
    th2 = output.state_variables['theta2']

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, th1, lw=0.8, label='θ₁', color='steelblue')
    ax.plot(t, th2, lw=0.8, label='θ₂', color='darkorange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')
    ax.set_title('Double pendulum — chaotic motion')
    ax.legend()
    fig.tight_layout()
    _save('DoublePendulum')


def fig_DampedSpring():
    from paleobeasts.signal_models.damped_spring import DampedSpring

    model = DampedSpring(forcing=None, m=1.0, k=4.0, c=0.4)
    output = model.integrate(t_span=(0, 30), y0=[1.0, 0.0], method='RK45')
    ts = output.to_pyleo(var_names=['x'])
    fig, ax = ts.plot(label='displacement (m)')
    ax.set_title('Damped spring — free oscillation')
    _save('DampedSpring')


# ---------------------------------------------------------------------------
# Climate models
# ---------------------------------------------------------------------------

def fig_Stommel():
    from paleobeasts.signal_models.stommel import Stommel

    model = Stommel(forcing=None, E=0.3, T_star=1.0, S_star=0.0)
    output = model.integrate(t_span=(0, 50), y0=[1.0, 0.0], method='RK45')
    ts_q = output.to_pyleo(var_names=['q'])
    fig, ax = ts_q.plot(label='q (overturning)')
    ax.set_title('Stommel — overturning strength')
    _save('Stommel')


def fig_Daisyworld():
    from paleobeasts.signal_models.daisyworld import Daisyworld

    model = Daisyworld(forcing=None, L=0.9)
    output = model.integrate(t_span=(0, 500), y0=[0.2, 0.2, 295.0], method='RK45')

    t = output.time
    Aw = output.state_variables['Aw']
    Ab = output.state_variables['Ab']
    T  = output.state_variables['T']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax1.plot(t, Aw, label='White daisies', color='lightblue')
    ax1.plot(t, Ab, label='Black daisies', color='dimgray')
    ax1.set_ylabel('Area fraction')
    ax1.legend(fontsize=8)
    ax2.plot(t, T, color='tomato')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_xlabel('Time')
    fig.suptitle('Daisyworld')
    fig.tight_layout()
    _save('Daisyworld')


def fig_ENSORechargeOscillator():
    from paleobeasts.signal_models.enso_recharge import ENSORechargeOscillator

    model = ENSORechargeOscillator(forcing=None, mu=0.75, Af=0.5, Pf=6.0)
    output = model.integrate(t_span=(0, 120), y0=[0.5, 0.0], method='RK45')
    ts = output.to_pyleo(var_names=['T'])
    fig, ax = ts.plot(label='SST anomaly (°C)')
    ax.set_title('ENSO recharge oscillator — SST anomaly')
    _save('ENSORechargeOscillator')


def fig_Model3():
    import paleobeasts as pb
    from paleobeasts.signal_models.g24 import Model3, calc_f

    orbital_forcing = pb.core.Forcing(calc_f)
    model = Model3(forcing=orbital_forcing)
    output = model.integrate(
        t_span=(-2000, 0), y0=[0.0, 1], method='RK45',
        kwargs={'max_step': 0.5},
    )
    ts = output.to_pyleo(var_names=['v'])
    fig, ax = ts.plot(label='Ice volume (normalised)')
    ax.set_title('Ganopolski (2024) Model 3 — glacial cycles')
    _save('Model3')


def fig_Stocker2003BipolarSeesaw():
    import paleobeasts as pb
    from paleobeasts.signal_models.stocker2003_bipolar_seesaw import (
        Stocker2003BipolarSeesaw,
    )

    Tn = pb.core.Forcing(lambda t: 1.0 if (t % 2000) < 1000 else -1.0)
    model = Stocker2003BipolarSeesaw(forcing=Tn, tau=500.0, beta=-1.0)
    output = model.integrate(t_span=(0, 8000), y0=[0.0], method='RK45')

    t = output.time
    Ts = output.state_variables['Ts']
    Tn_vals = output.diagnostic_variables['Tn']

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, Tn_vals, lw=1.0, label='T_N (north)', color='steelblue')
    ax.plot(t, Ts,      lw=1.0, label='T_S (south)', color='darkorange')
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('Temperature anomaly')
    ax.set_title('Stocker–Johnsen bipolar seesaw')
    ax.legend()
    fig.tight_layout()
    _save('Stocker2003BipolarSeesaw')


def fig_Stocker2003ExtendedSeaIceSeesaw():
    import paleobeasts as pb
    from paleobeasts.signal_models.stocker2003_bipolar_seesaw import (
        Stocker2003ExtendedSeaIceSeesaw,
    )

    T_N = pb.core.Forcing(lambda t: 1.0 if (t % 2000) < 1000 else 0.0)
    model = Stocker2003ExtendedSeaIceSeesaw(forcing=T_N)
    output = model.integrate(
        t_span=(0, 10000), y0=[0.0, 0.0, 0.3, 0.0], method='RK45'
    )
    ts = output.to_pyleo(var_names=['T_ANT'])
    fig, ax = ts.plot(label='Antarctic temperature anomaly')
    ax.set_title('Stocker extended seesaw — Antarctic temperature')
    _save('Stocker2003ExtendedSeaIceSeesaw')


def fig_TwoBoxCarbon():
    from paleobeasts.signal_models.two_box_carbon import TwoBoxCarbon

    model = TwoBoxCarbon(forcing=None, k=0.1, V_atm=1.0, V_surf=50.0)
    output = model.integrate(t_span=(0, 200), y0=[800.0, 38000.0], method='RK45')
    ts = output.to_pyleo(var_names=['A'])
    fig, ax = ts.plot(label='Atm. carbon inventory')
    ax.set_title('Two-box carbon — atmospheric inventory')
    _save('TwoBoxCarbon')


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def fig_downsample():
    import pyleoclim as pyleo
    from paleobeasts.utils.resample import downsample

    soi = pyleo.utils.load_dataset('SOI')
    soi_sparse = downsample(soi, method='exponential', param=[3.0], seed=42)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axes[0].plot(soi.time, soi.value, lw=0.6, color='steelblue')
    axes[0].set_ylabel('SOI')
    axes[0].set_title('Original (regular)')
    axes[1].plot(soi_sparse.time, soi_sparse.value, '.', ms=3, color='darkorange')
    axes[1].set_ylabel('SOI')
    axes[1].set_title('Downsampled (exponential gaps, scale=3)')
    axes[1].set_xlabel('Time')
    fig.tight_layout()
    _save('downsample')


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

FIGURES = {
    'EBM0D':                           fig_EBM0D,
    'EBM1DLat':                        fig_EBM1DLat,
    'Lorenz63':                        fig_Lorenz63,
    'Lorenz96':                        fig_Lorenz96,
    'Roessler':                        fig_Roessler,
    'SimplePendulum':                  fig_SimplePendulum,
    'DrivenPendulum':                  fig_DrivenPendulum,
    'DoublePendulum':                  fig_DoublePendulum,
    'DampedSpring':                    fig_DampedSpring,
    'Stommel':                         fig_Stommel,
    'Daisyworld':                      fig_Daisyworld,
    'ENSORechargeOscillator':          fig_ENSORechargeOscillator,
    'Model3':                          fig_Model3,
    'Stocker2003BipolarSeesaw':        fig_Stocker2003BipolarSeesaw,
    'Stocker2003ExtendedSeaIceSeesaw': fig_Stocker2003ExtendedSeaIceSeesaw,
    'TwoBoxCarbon':                    fig_TwoBoxCarbon,
    'downsample':                      fig_downsample,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', metavar='NAME',
                        help='generate figure for a single model only')
    parser.add_argument('--list', action='store_true',
                        help='list available model names and exit')
    args = parser.parse_args()

    if args.list:
        for name in sorted(FIGURES):
            print(f'  {name}')
        return

    targets = {args.model: FIGURES[args.model]} if args.model else FIGURES

    if args.model and args.model not in FIGURES:
        print(f'Unknown model: {args.model!r}. Use --list to see available names.')
        sys.exit(1)

    errors = []
    for name, func in targets.items():
        print(f'Generating {name}...')
        try:
            func()
        except Exception as exc:
            print(f'  ERROR: {exc}')
            errors.append((name, exc))

    print(f'\nDone. {len(targets) - len(errors)}/{len(targets)} figures generated.')
    if errors:
        print('Failed:')
        for name, exc in errors:
            print(f'  {name}: {exc}')
        sys.exit(1)


if __name__ == '__main__':
    main()
