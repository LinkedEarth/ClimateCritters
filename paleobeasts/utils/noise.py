"""Noise generation and surrogate time series utilities.

Thin wrappers around :class:`pyleoclim.SurrogateSeries` that expose surrogate
and parametric noise generation through a consistent PaleoBeasts interface.
"""

__all__ = [
    'from_series',
    'from_param',
]


def from_series(target_series, method, number=1, seed=None, label=None):
    """Generate surrogate series matched to a target Pyleoclim series.

    Parameters
    ----------
    target_series : pyleoclim.Series
        Target series used to infer surrogate properties (time axis,
        autocorrelation structure, etc.).
    method : str
        Surrogate method.  Supported values: ``'ar1sim'``, ``'phaseran'``,
        ``'uar1'``.
    number : int
        Number of surrogate realizations to generate.  Default 1.
    seed : int or None
        Random seed for reproducibility.  Default ``None``.
    label : str or None
        Label attached to the returned ``SurrogateSeries``.  Default ``None``.

    Returns
    -------
    surr : pyleoclim.SurrogateSeries
        Surrogate series object; ``surr.series_list`` contains ``number``
        series.

    See also
    --------
    pyleoclim.SurrogateSeries : Underlying surrogate generator.

    Examples
    --------
    .. code-block:: python

        import pyleoclim as pyleo
        from paleobeasts.utils.noise import from_series

        soi = pyleo.utils.load_dataset('SOI')
        surr = from_series(soi, method='ar1sim', number=10, seed=42)
    """
    import pyleoclim as pyleo
    surr = pyleo.SurrogateSeries(method=method, number=number, seed=seed, label=label)
    surr.from_series(target_series=target_series)
    return surr


def from_param(method='uar1', noise_param=None, length=50, number=1,
               time_pattern='even', settings=None, seed=None, label=None):
    """Generate surrogate series from a parametric noise model.

    Parameters
    ----------
    method : str
        Noise model.  Supported values: ``'ar1sim'``, ``'uar1'``,
        ``'CN'`` (colored noise).  Default ``'uar1'``.
    noise_param : list or None
        Model parameters:

        - ``'ar1sim'`` / ``'uar1'``: ``[tau, sigma0]``
        - ``'CN'``: ``[beta]``

        Default ``[1, 1]``.
    length : int
        Length of each surrogate series.  Default 50.
    number : int
        Number of surrogate realizations to generate.  Default 1.
    time_pattern : str
        Time-axis generation pattern.  One of:

        - ``'even'`` — evenly spaced with spacing ``delta_t`` from
          ``settings`` (default 1.0)
        - ``'random'`` — random spacing via ``delta_t_dist`` and ``param``
          in ``settings``
        - ``'specified'`` — explicit ``time`` array passed in ``settings``
    settings : dict or None
        Additional options forwarded to the surrogate generator.
        Default ``None``.
    seed : int or None
        Random seed for reproducibility.  Default ``None``.
    label : str or None
        Label attached to the returned ``SurrogateSeries``.  Default ``None``.

    Returns
    -------
    surr : pyleoclim.SurrogateSeries
        Surrogate series object; ``surr.series_list`` contains ``number``
        series.

    See also
    --------
    pyleoclim.SurrogateSeries : Underlying surrogate generator.

    Examples
    --------
    .. code-block:: python

        from paleobeasts.utils.noise import from_param

        # Ten AR(1) realizations with tau=5, sigma=0.5
        surr = from_param(method='ar1sim', noise_param=[5, 0.5],
                          length=200, number=10, seed=0)
    """
    if noise_param is None:
        noise_param = [1, 1]
    import pyleoclim as pyleo
    surr = pyleo.SurrogateSeries(method=method, number=number, seed=seed, label=label)
    surr.from_param(param=noise_param, length=length,
                    time_pattern=time_pattern, settings=settings)
    return surr
