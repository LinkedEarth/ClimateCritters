"""
Module Name: noise
Description: This module provides functions for generating and manipulating noise signals.
Author: Alexander James
"""

__all__ = [
    'from_series',
    'from_param'
]

# Import necessary modules
import warnings

import numpy as np

def from_series(target_series,method,number=1,seed=None,label=None):
    '''
    Fashion the SurrogateSeries object after a target series. 
    Methods that can be used here are 'ar1sim', 'phaseran', 'uar1', 'CN' (colored noise)

    Parameters
    ----------
    target_series : Series object
        target Series used to infer surrogate properties

    method : str
        surrogate method to use. Supported methods are: 'ar1sim', 'phaseran', 'uar1'

    number : int
        Number of surrogate series to generate. Default: 1
        
    seed : int
        Control random seed option for reproducibility

    label : str
        Label for the noise series.

    Returns
    -------
    surr : SurrogateSeries
        SurrogateSeries object. surr.series_list will contain as many series objects as specified by `number`

    See also
    --------

    pyleoclim.utils.tsmodel.ar1_sim : AR(1) simulator
    pyleoclim.utils.tsmodel.uar1_sim : maximum likelihood AR(1) simulator
    pyleoclim.utils.tsutils.phaseran2 : phase randomization
    
    Examples
    --------
    
    SOI = pyleo.utils.load_dataset('SOI')
    noise = from_series(SOI,method='ar1sim')

    '''
    import pyleoclim as pyleo
    surr = pyleo.SurrogateSeries(method=method,number=number,seed=seed,label=label)
    surr.from_series(target_series=target_series)
    return surr

    #settings = {} if settings is None else settings.copy()
        
    # if seed is not None:
    #     np.random.default_rng(seed)
    
    # # apply surrogate method
    # if method == 'ar1sim':
    #         y_noise = pyleo.utils.tsmodel.ar1_sim(target_series.value, number, target_series.time) * scale

    # elif method == 'phaseran':
    #     if target_series.is_evenly_spaced():
    #         y_noise = pyleo.utils.tsutils.phaseran2(target_series.value, number) * scale
    #     else:
    #         raise ValueError("Phase-randomization presently requires evenly-spaced series.")

    # elif method == 'uar1':
    #     # estimate theta with MLE
    #     tau, sigma_2 = pyleo.utils.tsmodel.uar1_fit(target_series.value, target_series.time)
    #     # generate time axes according to provided pattern
    #     times = np.squeeze(np.tile(target_series.time, (number, 1)).T)  
    #     # generate matrix
    #     y_noise = pyleo.utils.tsmodel.uar1_sim(t = times, tau=tau, sigma_2=sigma_2) * scale
    
    # else:
    #     raise ValueError('Noise method not recognized')
    
    # if number > 1:
    #     s_list = []
    #     for i, y in enumerate(y_noise.T):
    #         ts = target_series.copy() # copy Series
    #         ts.value += y.flatten() # replace value with y_noise column
    #         ts.label = str(target_series.label or '') + " surr #" + str(i+1)
    #         s_list.append(ts)
    #     ens = pyleo.EnsembleSeries(s_list)
    #     return ens
    # else:
    #     ts = target_series.copy() # copy Series
    #     ts.value += y_noise.flatten() # replace value with y_noise column
    #     ts.label = str(target_series.label or '') + " surr" 
    #     return ts
                  
def from_param(method = 'uar1',noise_param=[1,1],length=50, number = 1, time_pattern = 'even', settings=None,seed=None,label=None):
    '''
    Simulate the noise realizations from a parametric model.
    Note that if you wish to use a custom time axis, set time_pattern to be 'specified' you can pass it in the settings dictionary with the key 'time'.

    Parameters
    ----------

    method : str
        surrogate method to use. Supported methods are: 'ar1sim', 'uar1', 'CN' (colored noise)

    noise_param : list
        model parameters (e.g. [tau, sigma0] for an AR(1) model)
        * AR(1) model ['uar1','ar1sim']: [tau, sigma0]
        * CN: [beta]
    
    length : int
        Length of the series. Default: 50

    number : int
        Number of noise realizations to generate. Default: 1
        
    time_pattern : str {even, random, specified}
        The pattern used to generate the surrogate time axes
        * 'even' uses an evenly-spaced time with spacing `delta_t` specified in settings (if not specified, defaults to 1.0)
        * 'random' uses random_time_axis() with specified distribution and parameters 
            Relevant settings are `delta_t_dist` and `param`. (default: 'exponential' with parameter = 1.0)
        * 'specified': uses time axis `time` from `settings`.  

    settings : dict
        Parameters for surrogate generator. See individual methods for details. 
        
    seed : int
        Control random seed option for reproducibility

    label : str
        Label for the noise series.

    Returns
    -------
    surr : SurrogateSeries
        SurrogateSeries object. surr.series_list will contain as many series objects as specified by `number`

    See also
    --------

    pyleoclim.utils.tsmodel.ar1_sim : AR(1) simulator
    pyleoclim.utils.tsmodel.uar1_sim : maximum likelihood AR(1) simulator
    pyleoclim.utils.tsmodel.random_time_index : Generate time increment vector according to a specific probability model
    
    Examples
    --------

    noise = from_param(method='ar1sim',noise_param=[1,1])

    '''
    import pyleoclim as pyleo
    surr = pyleo.SurrogateSeries(method=method,number=number,seed=seed,label=label)
    surr.from_param(param=noise_param,length=length,time_pattern=time_pattern,settings=settings)
    return surr
    # noise_param = list(noise_param) if noise_param is not None else [] # coerce param into a list, no matter the original format
    # nparam = len(noise_param)
    
    # settings = {} if settings is None else settings.copy()
    
    # if seed is not None:
    #     np.random.seed(seed)
    
    # # generate time axes according to provided pattern
    # if time_pattern == "even":
    #     delta_t = settings["delta_t"] if "delta_t" in settings else 1.0
    #     t = np.cumsum([float(delta_t)]*length)
    #     times = np.tile(t, (number, 1)).T     
    # elif time_pattern == "random":
    #     times = np.zeros((length, number))
    #     for i in range(number):
    #         dist_name = settings['delta_t_dist'] if "delta_t_dist" in settings else "exponential"
    #         dist_param = settings['param'] if "param" in settings else [1]
    #         times[:, i] = pyleo.utils.tsmodel.random_time_axis(length, dist_name,dist_param) 
    # elif time_pattern == 'specified':
    #     if "time" not in settings:
    #         raise ValueError("'time' not found in settings")
    #     else:
    #         times =  np.tile(settings["time"], (number, 1)).T 
    # else:
    #     raise ValueError(f"Unknown time pattern: {time_pattern}")
           
    # n = times.shape[0]
    
    # if method in ['uar1','ar1sim']: #
    #     if nparam<2:
    #         warnings.warn(f'The AR(1) model needs 2 parameters, tau and sigma2 (in that order); {nparam} provided. default values used, tau=5, sigma=0.5',UserWarning, stacklevel=2)
    #         noise_param = [5,0.5]
    #     ar1_vectors = []
    #     for i in range(number):
    #         ar1_noise = pyleo.utils.tsmodel.ar1_model(t = times.T[i], tau=noise_param[0], output_sigma=noise_param[1])
    #         ar1_noise = ar1_noise * (scale/np.std(ar1_noise))
    #         ar1_vectors.append(ar1_noise)
    #     y_noise = np.column_stack(ar1_vectors)

    # #Note I don't know if these methods (power law and fgn) are properly implemented because i made them with claude.
    # elif method == 'power_law':
    #     beta = noise_param[0]
    #     # Generate the frequencies
    #     freqs = np.fft.rfftfreq(n)

    #     # Generate the power spectrum
    #     power_spectrum = np.zeros_like(freqs)
    #     power_spectrum[freqs == 0] = 1.0  # DC component
    #     power_spectrum[freqs > 0] = np.power(freqs[freqs > 0], -beta / 2)

    #     colored_noise_vectors = []
    #     for _ in range(number):
    #         # Generate random phases
    #         phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))
    #         phases[0] = 1  # DC component has no phase

    #         # Construct the frequency-domain representation
    #         freq_domain = power_spectrum * phases

    #         # Generate the colored noise by taking the inverse Fourier transform
    #         colored_noise = np.fft.irfft(freq_domain, n=n)

    #         # Scale the noise to have the desired standard deviation
    #         colored_noise = colored_noise * (scale / np.std(colored_noise))

    #         colored_noise_vectors.append(colored_noise)

    #     y_noise = np.column_stack(colored_noise_vectors)

    # elif method == 'fGn':
    #     # Generate the first row of the circulant matrix
    #     if nparam > 1:
    #         warnings.warn(f'The fGn model needs 1 parameters, H (the Hurst exponent); {nparam} provided. default values used, H=0.8',UserWarning, stacklevel=2)
    #         noise_param = [0.8]
    #     elif noise_param[0] < 0 or noise_param[0] > 1:
    #         raise ValueError("The Hurst exponent must be between 0 and 1.")
    #     k = np.arange(1, n)
    #     H = noise_param[0]
    #     C = (abs(k - 1) ** (2 * H) - 2 * k ** (2 * H) + (k + 1) ** (2 * H)) / 2

    #     # Generate the eigenvalues of the circulant matrix
    #     lambdas = np.sqrt(np.concatenate(([0], C[:n//2], C[:(n-1)//2:-1])))

    #     fgn_vectors = []
    #     for _ in range(number):
    #         # Generate random phases
    #         phi = np.random.uniform(0, 2 * np.pi, n)

    #         # Construct the fractional Gaussian noise
    #         Z = np.fft.fft(lambdas * np.exp(1j * phi))
    #         fgn = np.real(Z[:n])

    #         # Scale the noise to have the desired standard deviation
    #         fgn = fgn * (scale / np.std(fgn))

    #         fgn_vectors.append(fgn)

    #     y_noise = np.column_stack(fgn_vectors)

    # elif method == 'white':
    #     # Generate white noise with zero mean and unit variance
    #     white_noise = np.random.normal(0, 1, (n, number))

    #     # Scale the noise to have the desired standard deviation
    #     white_noise = white_noise * scale

    #     y_noise = white_noise
    
    # # create the series_list    
    # if number > 1:
    #     s_list = []
    #     for i, (t, y) in enumerate(zip(times.T,y_noise.T)):
    #         ts = pyleo.Series(time=t, value=y,  
    #                         label = "Noise #" + str(i+1),
    #                         verbose=False, auto_time_params=True)
    #         s_list.append(ts)
    #     ens = pyleo.EnsembleSeries(s_list)
    #     return ens

    # else:
    #     ts = pyleo.Series(time=times.flatten(), value=y_noise.flatten(), 
    #             label = "Noise",
    #             verbose=False, auto_time_params=True)
    #     return ts