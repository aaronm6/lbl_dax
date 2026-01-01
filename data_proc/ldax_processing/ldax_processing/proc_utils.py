"""
Utilities for processing that don't fall under C extension code or filtering or file reading
"""
import numpy as np

def bs_subtract_first_n(d, n=100):
    """
    Returns a copy of d with baseline from each waveform subtracted.  Baseline is determined
    by averaging the first n samples in each waveform.
    
    Inputs:
    d : n_evt x n_ch x n_samples (i.e. full dataset) MUST BE DTYPE=FLOAT
    n : number of samples at the start of each waveform to use as the baseline average
    
    Returns:
    new d (with baseline subtracted)
    """
    if not np.issubdtype(d.dtype, np.floating):
        raise TypeError("input d's dtype must be floating point")
    d_bs = d[...,:n].mean(axis=-1)
    return d - d_bs[..., np.newaxis]

def bs_subtract_nonpulse_avg(d, pulse_mask):
    """
    Returns a copy of d with baseline from each waveform subtracted.  Baseline is determined
    by using the pulse_mask input to look at only the regions of each waveform where there
    are no pulses, and averages those.  This does a better job than bg_subtract_first_n
    in that it (in principle) uses info from the whole waveform.
    
    Inputs:
    d : n_evt x n_ch x n_samples -> a full data set: MUST BE DTYPE=FLOAT
    pulse_mask : same size as d; boolean array, true where there are pulses
    
    Returns:
    new d (with baseline subtracted)
    """
    if not np.issubdtype(d.dtype, np.floating):
        raise TypeError("input d's dypte must be floating")
    d_noise_ma = np.ma.masked_array(d, mask=pulse_mask)
    d_bs = d_noise_ma.mean(axis=-1).data
    return d - d_bs[..., np.newaxis]

def bs_subtract_nonpulse_poly(d, pulse_mask, order=7):
    """
    Returns a copy of d with baseline from each waveform subtracted.  Each waveform,
    after cutting out pulses, is fit with a n'th order polynomial.  This polynomial
    is then subtracted from its waveform, so it can account for low-frequency wiggles
    in the baseline level.  Polynomial fits can be vectorized, but only whent he 
    x-vector is the same; this means that we cannot perform a vectorized version
    of this operation over an entire data set at once; we instead have to do 
    it event-by-event (but within an event, all 32 channels can be vectorized).
    
    Inputs:
    d : n_evt x n_ch x n_samples -> a full data set: MUST BE DTYPE=FLOAT
    pulse_mask : same size as d; boolean array, true where there are pulses
    n : Order of the polynomial (default is 7).
    
    Returns:
    new d (with baseline subtracted).
    """
    if not np.issubdtype(d.dtype, np.floating):
        raise TypeError("input d's dypte must be floating")
    n_evt, n_ch, n_samp = d.shape
    d_new = np.empty_like(d)
    x = np.linspace(-1.,1.,n_samp)
    for k in range(n_evt):
        x_cut = ~(pulse_mask[k,...].any(axis=0))
        p_coeffs = np.polyfit(x[x_cut], d[k,:,x_cut], order)
        x_mat = x[:, np.newaxis] ** (np.r_[order:-1:-1][np.newaxis,:])
        bs_curves = (x_mat[..., np.newaxis] * p_coeffs[np.newaxis,...]).sum(axis=1)
        d_new[k,...] = d[k,...] - bs_curves.T
    return d_new

bs_subtract_dict = {
    'avg_first_n' : bs_subtract_first_n,
    'avg_nonpulse' : bs_subtract_nonpulse_avg,
    'polyfit_nonpulse' : bs_subtract_nonpulse_poly}

def bs_subtract(d, *args, method='avg_first_n', **kwargs):
    """
    baseline subtraction.  See bs_subtract_dict for 'method' options, and
    each value in that dict has a docstring to check for documentation.
    
    Input (common for any method):
    d : n_evt x n_ch x n_samples (3d) numpy array, dtype must be a float
    Other inputs: check docstring for desired method
    
    Returns :
    d_new : a version of d where the baseline has been subtracted from the waveform
            of every channel and every event.
    """
    if method not in bs_subtract_dict:
        raise ValueError("method must be a string that matches a key in bs_subtract_dict")
    if method in ('avg_nonpulse','polyfit_nonpulse') and not args:
        raise ValueError(f"pulse_mask must be given if method={method} is given")
    return bs_subtract_dict[method](d, *args, **kwargs)



