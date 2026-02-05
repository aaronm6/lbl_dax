import numpy as np
import ldax_processing as ldax
import varray as va

__all__ = [
    'baseline_rolling',
    'baseline_avg_1st_n',
    'filter_lowpass_RC',
    'pod_bool_per_ch',
    'pod_bool_sum_ch',
    'calc_p_bnds']

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

def pod_bool_per_ch(d_filt, c):
    """
    Create a per-channel-per-event boolean that is True where there are pods
    """
    # find pods, creating a boolean per-channel array that identifies pods
    pod_bool = ldax.pod_boolean(d_filt, 
        thresh=c.ch_podbool_thresh, 
        prepod_samples=c.ch_podbool_prepodsamples, 
        postpod_samples=c.ch_podbool_postpodsamples)
    return pod_bool

def pod_bool_sum_ch(d_chsum, c):
    """
    Create a pod boolean based on the per-event sum over channels
    """
    d_chsum_pod = ldax.pod_boolean(d_chsum, 
        thresh=c.sm_podbool_thresh, 
        prepod_samples=c.sm_podbool_prepodsamples, 
        postpod_samples=c.sm_podbool_postpodsamples)
    ldax.merge_islands(d_chsum_pod, width=c.sm_merge_islands_width)
    # Make sure the start and end of the waveform are not part of a pulse
    d_chsum_pod[:,0] = False
    d_chsum_pod[:,-1] = False
    
    return d_chsum_pod

def calc_p_bnds(d_chsum,d_chsum_pod, c):
    """
    Take the boolean area (of waveforms summed over channel) and produce a varray of
    pulse boundaries.  The varray is of size (num_evt x 2 x num_pulses), where the
    middle dimension is of size 2 because it is a doublet of start_index, stop_index
    for each pulse found in each event.
    """
    # Get the pod starts and stops -- this will form the first guess at the pulse boundaries.
    p_starts_evt, p_starts_samp = np.nonzero(np.diff(d_chsum_pod.astype(np.int8),axis=-1)==1)
    p_stops_evt, p_stops_samp = np.nonzero(np.diff(d_chsum_pod.astype(np.int8),axis=-1)==-1)
    
    # check that there are the same number of starts and stops in each event
    if len(p_starts_evt) != len(p_stops_evt):
        raise ValueError("The number of pulse starts and stops must be the same in each event")
    if not (p_starts_evt == p_stops_evt).all():
        raise ValueError("The number of pod starts and stops is not the same")
    
    # calculate the boundaries of each pulse
    pulse_sarray = np.empty(d_chsum_pod.shape[0], dtype=np.uint16)
    for k in range(d_chsum_pod.shape[0]):
        n_pulses = (p_starts_evt==k).sum()
        pulse_sarray[k] = n_pulses
    p_bnds = va.varray(
        darray=np.vstack([p_starts_samp+1, p_stops_samp]), sarray=pulse_sarray, dtype=np.uint32)
    
    # Split off low-amplitude tails from pulses; needs to be done several times
    split_dict = {
        'amp_frac': c.split_amp_frac,
        'amp_max': c.split_amp_max,
        'quiet_samples': c.split_quiet_samples,
        'buffer_samples': c.split_buffer_samples}
    for k in range(c.split_iterations):
        p_bnds = va.varray(ldax.split_pulses(p_bnds.flatten(), p_bnds.sarray,d_chsum, **split_dict))
    
    return p_bnds
