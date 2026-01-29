import numpy as np
import ldax_processing as ldax

def baseline_rolling(d, c):
    """
    Determine the per-ch-per-event baseline estimate by a rolling average over periods
    without pulses.
    
    d is raw data (cast as np.float64)
    c is a dict_attr object with contents loaded from the settings yaml file
    """
    # compute pulse-finding filtered waveform, "d_find" and create boolean mask based on it
    d_find = ldax.lowpass_ngdd(d, c.find_ngdd_width)
    d_find_mask = ldax.ngdd_filt_mask(
        d_find,
        thresh=c.find_mask_thresh,
        pre_samples_add=c.find_mask_pre_samples_add,
        post_samples_add=c.find_mask_post_samples_add)
    ldax.merge_islands(d_find_mask, width=c.chfind_merge_islands_width)
    
    # calculate per-channel-per-event baseline curves from a running average, masked of pulses
    d_bs_est = ldax.baseline_update(d, d_find_mask, alpha=c.bs_est_alpha)
    return d_bs_est

def baseline_avg_1st_n(d, c):
    """
    Estimate the per-ch-per-event baseline as the average of the first n samples in
    each channel in each event.  n is given as 'bs_1st_n' in the yaml settings file.
    """
    d_bs_est = d[..., :c.bs_1st_n].mean(axis=-1)
    return d_bs_est[..., np.newaxis]

def filter_lowpass_RC(d, c):
    """
    Apply an n-pole RC filter to the data on a per-ch-per-event basis, specifying
    the overall filter bandwidth and the number of poles.
    """
    return ldax.lowpass_RC(d, c.filter_RC_bw, n=c.filter_RC_poles)

def pod_bool_calc(d_filt, c):
    """
    Create a per-channel-per-event boolean that is True where there are pods
    """
    # find pods, creating a boolean per-channel array that identifies pods
    pod_bool = ldax.pod_boolean(d_filt, 
        thresh=c.ch_podbool_thresh, 
        prepod_samples=c.ch_podbool_prepodsamples, 
        postpod_samples=c.ch_podbool_postpodsamples)
    return pod_bool
