"""Downsampling utilities for irregular paleoclimate time series."""

__all__ = [
    'downsample',
]

import numpy as np


def downsample(series, method='exponential', param=None, return_index=False, seed=None):
    """Downsample a Pyleoclim series by drawing random time increments.

    Simulates irregular sampling by generating random index increments from a
    chosen probability distribution and selecting the corresponding time points
    from the original series.

    Parameters
    ----------
    series : pyleoclim.Series
        The time series to downsample.
    method : str
        Probability distribution used to draw index increments.  One of:

        - ``'exponential'`` — exponential distribution; ``param`` is a
          1-element list ``[scale]`` (i.e. mean gap size).
        - ``'poisson'`` — Poisson distribution; ``param`` is ``[rate]``.
        - ``'pareto'`` — Pareto distribution; ``param`` is
          ``[shape, scale]``.
        - ``'random_choice'`` — discrete distribution; ``param`` is
          ``[values, probabilities]`` where both arrays have the same length.

        Default ``'exponential'``.
    param : list or None
        Parameter(s) for the chosen distribution.  Default ``[1]``
        (exponential with scale 1).
    return_index : bool
        If ``True``, return the integer index array instead of a new series.
        Default ``False``.
    seed : int or None
        Seed for the random number generator.  Pass an integer for
        reproducible results.  Default ``None``.

    Returns
    -------
    downsampled : pyleoclim.Series or list of int
        Downsampled series (``return_index=False``) or list of selected
        indices (``return_index=True``).

    Raises
    ------
    ValueError
        If ``method`` is not recognised, or ``param`` has the wrong shape
        for the chosen distribution.

    Examples
    --------
    ```python
    import matplotlib.pyplot as plt
    import pyleoclim as pyleo
    from paleobeasts.utils.resample import downsample

    soi = pyleo.utils.load_dataset('SOI')
    soi_sparse = downsample(soi, method='exponential', param=[3.0], seed=42)
    soi_sparse.plot()
    plt.savefig('docs/reference/figures/downsample_example.png',
                dpi=150, bbox_inches='tight')
    ```
    """
    if param is None:
        param = [1]

    valid_methods = ['exponential', 'poisson', 'pareto', 'random_choice']
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}.")

    rng = np.random.default_rng(seed)
    n = len(series.time)

    if method == 'exponential':
        p = np.asarray(param, dtype=float)
        if p.size != 1:
            raise ValueError("'exponential' requires a single scale parameter: param=[scale].")
        delta_t = rng.exponential(scale=float(p[0]), size=n)

    elif method == 'poisson':
        p = np.asarray(param, dtype=float)
        if p.size != 1:
            raise ValueError("'poisson' requires a single rate parameter: param=[rate].")
        delta_t = rng.poisson(lam=float(p[0]), size=n) + 1

    elif method == 'pareto':
        p = np.asarray(param, dtype=float)
        if p.size != 2:
            raise ValueError("'pareto' requires shape and scale parameters: param=[shape, scale].")
        delta_t = (rng.pareto(float(p[0]), size=n) + 1) * float(p[1])

    elif method == 'random_choice':
        values = np.asarray(param[0])
        probs = np.asarray(param[1], dtype=float)
        if len(values) != len(probs):
            raise ValueError(
                "'random_choice' requires param=[values, probabilities] "
                "where both arrays have the same length."
            )
        delta_t = rng.choice(values, size=n, p=probs)

    # Enforce minimum gap of 1 index step
    delta_t_int = [max(1, int(d)) for d in delta_t]

    # Cumulative index; keep only those within bounds
    index = [v for v in np.cumsum(delta_t_int) if v < n]

    if return_index:
        return index

    new_series = series.copy()
    new_series.time = series.time[index]
    new_series.value = series.value[index]
    return new_series
