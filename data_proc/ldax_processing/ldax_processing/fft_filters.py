import numpy as np

def _RC_n_filt(freq, bw, n=1):
    g_out = 1/((freq*((2**(1/n))-1)/bw + 1)**n)
    return g_out

def _exp_filt(freq, bw):
    g_out = np.exp(-freq*np.log(2)/bw)
    return g_out

def _lowpass_util(y, bw, axis=-1, filt_func=_RC_n_filt, **kwargs):
    assert isinstance(y, np.ndarray), "y must be a numpy array"
    assert y.ndim in (1,2), "Input y must be 1d or 2d"
    
    Y = np.fft.rfft(y, axis=axis)
    freq = np.r_[0:0.5:(1j)*Y.shape[axis]]
    if (axis==0) and (Y.ndim==2):
        freq = freq[:,np.newaxis]
    gain = filt_func(freq, bw, **kwargs)
    Y_filt = Y * gain
    y_out = np.fft.irfft(Y_filt, axis=axis)
    return y_out

def lowpass_RC(y, bw, n=1, axis=-1):
    """
    Perform an n-pole RC-like low-pass filter of the data contained in y.
    Inputs:
          y: the 1d or 2d numpy array containing the time-series data to be filtered.
         bw: The bandwidth of the filter, in units of 1/samples.  So if your sampling interval
             were 10ns (i.e. 100 MS/s) and you wanted to implement a filter at 20 MHz, your
             bw here would be 0.2.  Note that the bw is the frequency at which the gain
             is 0.5.
          n: The number of poles of the filter.  That is, n=1 means just a simple RC filter.
             n=2 means there are two RC filters in series, etc.  Note that as n gets large,
             the result of the filter approaches that of the exponential filter.  Also note
             that if I just naively repeat an RC filter twice, the bandwidth of the new filter
             is NOT 1/(2*pi*RC) anymore.  But here when n > 1, the bw parameter does still 
             actually describe the true bandwidth of the filter (i.e. the frequency at which the
             gain of the filter is 0.5).
       axis: The axis over which the filter will be performed.  If axis=1, then each row is
             treated as a separate waveform; if axis=0, then each column is treated as a 
             separate waveform.  The default is axis=-1, which just means it would take the last
             dimension (i.e. axis=1 if 2d).
    Outputs:
     y_filt: The filtered form of the dat in y.
    """
    return _lowpass_util(y, bw, axis=axis, filt_func=_RC_n_filt, n=n)

def lowpass_exp(y, bw, axis=-1):
    """
    Perform an exponential low-pass filter of the data contained in y.  That is, the fft
    of the waveform is scaled with a decaying exponential.
    Inputs:
          y: the 1d or 2d numpy array containing the time-series data to be filtered.
         bw: The bandwidth of the filter, in units of 1/samples.  So if your sampling interval
             were 10ns (i.e. 100 MS/s) and you wanted to implement a filter at 20 MHz, your
             bw here would be 0.2.  Note that the bw is the frequency at which the gain
             is 0.5.
       axis: The axis over which the filter will be performed.  If axis=1, then it is assumed 
             that each row is a waveform.  The default is axis=-1, which just means it would
             take the last dimension (i.e. axis=1 if 2d).
    Outputs:
     y_filt: The filtered form of the dat in y.
    """
    return _lowpass_util(y, bw, axis=axis, filt_func=_exp_filt)

#forwardconv(s_raw, s_kern, axis=-1)\n--\n\n"
def forwardconv_fft(s_raw, s_kern, axis=-1):
    """
    The same as the c-compiled forward-convolution, but here the calculation is performed in 
    Fourier space.  length of s_kern must be less than that of s_raw
    """
    s_kern_pad = np.r_[s_kern, zeros(s_raw.shape[axis]-len(s_kern))]
    S_raw = fft.rfft(s_raw, axis=axis)
    S_kern = fft.rfft(s_kern_pad)
    if (axis % A.ndim) == 0:
        S_kern_broadcast = S_kern[:, newaxis]
    else:
        S_kern_broadcast = S_kern[newaxis,:]
    F_fc = S_raw * (S_kern_broadcast.conjugate())
    f_fc = fft.irfft(F_fc)
    return f_fc










