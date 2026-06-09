import matplotlib.pyplot as plt
import numpy as np

def plot_solvers(solvers, forcing_func, params, time_ranges=None, sampling_res=1, plot_solver_time=True, labels=None):
    n_solvers = len(solvers)
    fig, ax = plt.subplots(n_solvers + 1, 1, sharex=True, figsize=(10, 8))

    # If no custom time ranges are given, use the solver's time range
    if time_ranges is None:
        time_ranges = [np.arange(solver.solution.t[0], solver.solution.t[-1], sampling_res) for solver in solvers]

    # Plot forcing functions
    for i, solver in enumerate(solvers):
        label = labels[i] if labels and i < len(labels) else f'Solver {i + 1}'

        if plot_solver_time:
            time = solver.solution.t
            forcing = forcing_func(time, *params)
            ax[0].plot(time, forcing, label=f'{label} (solver time)')

        if time_ranges[i] is not None:
            forcing = forcing_func(time_ranges[i], *params)
            ax[0].plot(time_ranges[i], forcing, label=f'{label} (custom time)', linestyle='--')

    ax[0].set_ylabel('Forcing')
    ax[0].legend()

    # Plot each solver's solution
    for i, solver in enumerate(solvers):
        if plot_solver_time:
            ax[i + 1].plot(solver.solution.t, solver.solution.y[0], label=f'{label} (solver time)', marker='o',
                           markersize=2)

        if time_ranges[i] is not None:
            sol_interp = np.interp(time_ranges[i], solver.solution.t, solver.solution.y[0])
            ax[i + 1].plot(time_ranges[i], sol_interp, label=f'{label} (custom time)', linestyle='--')

        ax[i + 1].set_ylabel(f'{label} Ice Volume')
        ax[i + 1].legend()

    ax[-1].set_xlabel('Time (kyr)')
    plt.tight_layout()
    plt.show()