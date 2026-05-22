"""PBOutput — container for a single PBModel integration run.

Separating run output from model configuration lets the same model instance
be re-run with different parameters or initial conditions while keeping each
run's results independently accessible.
"""

from __future__ import annotations

import numpy as np


class PBOutput:
    """Container for the results of one call to ``PBModel.integrate()``.

    ``PBOutput`` carries the full trajectory produced by the solver and
    exposes output-focused operations (noise addition, pyleoclim export,
    time resampling).  Keeping these on the output rather than on the model
    means a single model instance can produce multiple independent outputs
    without them overwriting each other.

    Attributes
    ----------
    model_time : ndarray of float
        Raw time axis as returned by the solver.  Never modified after
        construction; inspect this to understand the solver's actual grid.
    time : ndarray of float
        User-facing time axis.  Starts equal to ``model_time``; replaced by
        ``reframe_time_axis`` when a different grid is requested.
        ``state_variables`` is always aligned to ``time``.
    state_variables : structured ndarray
        Named state variable arrays, indexed by variable name.
    state_variable_names : list of str
        Ordered list of state variable names.
    diagnostic_variables : dict of str → ndarray
        Named diagnostic arrays.
    solution : object
        Raw solver output (retained for dense-output resampling via
        ``reframe_time_axis``).
    run_name : str
        Label for this integration run.
    """

    def __init__(
        self,
        time,
        state_variables,
        state_variable_names,
        diagnostic_variables,
        solution,
        run_name,
    ):
        self.model_time = np.asarray(time, dtype=float)
        self.time = self.model_time          # same object until reframe_time_axis
        self.state_variables = state_variables
        self.state_variable_names = list(state_variable_names)
        self.diagnostic_variables = diagnostic_variables
        self.solution = solution
        self.run_name = str(run_name) if run_name is not None else None
        self._noise_originals = {}
        self._noisy_vars = set()

    # ------------------------------------------------------------------
    # Variable lookup
    # ------------------------------------------------------------------

    def get_series_by_name(self, var_name):
        """Return a variable's array and its storage location.

        State variables live in a structured numpy array; diagnostic
        variables live in a dict.  Returning the location string lets
        callers that write back (e.g. ``add_noise``) know which container
        to update.

        Returns
        -------
        values : ndarray of float
        location : {'state', 'diagnostic'}
        """
        if (
            self.state_variables is not None
            and self.state_variable_names
            and var_name in self.state_variable_names
        ):
            return np.asarray(self.state_variables[var_name], dtype=float), "state"
        if var_name in self.diagnostic_variables:
            return np.asarray(self.diagnostic_variables[var_name], dtype=float), "diagnostic"
        raise ValueError(f"'{var_name}' not found in state variables or diagnostics.")

    # ------------------------------------------------------------------
    # Pyleoclim export
    # ------------------------------------------------------------------

    def to_pyleo(self, var_names=None):
        """Export one or more variables as pyleoclim Series objects.

        Parameters
        ----------
        var_names : str or list of str
            Name(s) of state or diagnostic variable(s) to export.

        Returns
        -------
        pyleoclim.Series or pyleoclim.MultipleSeries
        """
        from pyleoclim.core import Series, MultipleSeries

        if isinstance(var_names, str):
            var_names = [var_names]

        pyleo_series = []
        for var_name in var_names:
            if var_name in self.state_variable_names:
                value = self.state_variables[var_name]
            elif var_name in self.diagnostic_variables:
                value = self.diagnostic_variables[var_name]
            else:
                raise ValueError(
                    f"'{var_name}' not found. "
                    "Check state_variable_names and diagnostic_variables."
                )

            time = self.time
            value = np.asarray(value)
            if len(time) != len(value):
                n = min(len(time), len(value))
                time = time[:n]
                value = value[:n]

            pyleo_series.append(
                Series(
                    time=time,
                    value=value,
                    value_name=var_name,
                    verbose=False,
                    auto_time_params=True,
                )
            )

        return pyleo_series[0] if len(pyleo_series) == 1 else MultipleSeries(pyleo_series)

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def add_noise(self, var_name, noise_ts):
        """Add externally provided noise to an output variable.

        The unmodified values are saved on the first call so that
        ``remove_noise`` can restore the clean series.

        Parameters
        ----------
        var_name : str
            Name of a state or diagnostic variable.
        noise_ts : array-like
            Noise series with the same shape as the target variable.
        """
        values, location = self.get_series_by_name(var_name)
        noise_arr = np.asarray(noise_ts, dtype=float)
        if noise_arr.shape != values.shape:
            raise ValueError(
                f"Noise shape {noise_arr.shape} does not match "
                f"variable shape {values.shape} for '{var_name}'."
            )
        if var_name not in self._noise_originals:
            self._noise_originals[var_name] = values.copy()

        noisy = values + noise_arr
        if location == "state":
            self.state_variables[var_name] = noisy
        else:
            self.diagnostic_variables[var_name] = noisy
        self._noisy_vars.add(var_name)

    def remove_noise(self, var_name):
        """Restore a variable to its pre-noise values.

        Reverses ``add_noise`` by replacing the noisy array with the
        clean copy saved on the first ``add_noise`` call.
        """
        if var_name not in self._noise_originals:
            raise ValueError(f"No stored clean version for '{var_name}'.")
        original = self._noise_originals[var_name]
        _, location = self.get_series_by_name(var_name)
        if location == "state":
            self.state_variables[var_name] = original
        else:
            self.diagnostic_variables[var_name] = original
        self._noise_originals.pop(var_name, None)
        self._noisy_vars.discard(var_name)

    # ------------------------------------------------------------------
    # Time resampling
    # ------------------------------------------------------------------

    def reframe_time_axis(self, t_eval):
        """Resample state variables onto a new time axis.

        Uses the dense output from ``solve_ivp`` when available (accurate
        polynomial interpolation); falls back to linear interpolation for
        fixed-step solvers.

        After this call, ``time`` is replaced by ``t_eval`` and
        ``state_variables`` is replaced by the interpolated values.
        ``model_time`` is never modified and always reflects the solver's
        original grid.

        Parameters
        ----------
        t_eval : array-like
            Target time axis.

        Returns
        -------
        reframed : structured ndarray or ndarray
            Resampled state variables on ``t_eval``.
        """
        if self.solution is None:
            raise ValueError("No solution stored in this output.")

        t_eval = np.asarray(t_eval, dtype=float)

        if hasattr(self.solution, 'sol') and self.solution.sol is not None:
            y_eval = self.solution.sol(t_eval).T
        else:
            t_src = np.asarray(self.solution.t, dtype=float)
            y_src = np.asarray(self.solution.y, dtype=float)
            if y_src.ndim == 1:
                y_src = y_src.reshape(-1, 1)
            y_eval = np.column_stack([
                np.interp(t_eval, t_src, y_src[:, i])
                for i in range(y_src.shape[1])
            ])

        if self.state_variable_names:
            dtype = [(var, float) for var in self.state_variable_names]
            reframed = np.zeros(len(t_eval), dtype=dtype)
            for i, var in enumerate(self.state_variable_names):
                reframed[var] = y_eval[:, i]
        else:
            reframed = y_eval

        self.time = t_eval
        self.state_variables = reframed
        return reframed
