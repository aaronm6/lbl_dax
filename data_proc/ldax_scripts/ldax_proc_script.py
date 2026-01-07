import os, sys
import numpy as np
import ldax_processing as ldax
import varray as va
import tracemalloc

raw_data_path = '/mnt/drive1/TPC_data'
rq_path = '/mnt/drive2/TPC_RQs'

def process_portion(filename_and_path, start_event, num_events):
    # load data
    d, _, _ = ldax.Read_DDC40_fName(filename_and_path, start_event=start_event, num_events=num_events)
    
    # convert waveform from dtype=int16 to dtype=float64
    d = d.astype(float)
    
    # compute pulse-finding filtered waveform, "d_find" and create boolean mask based on it
    d_find = ldax.lowpass_ngdd(d, 0.15)
    d_find_mask = ldax.ngdd_filt_mask(
        d_find,
        thresh=3.5/.6, #3.2/.6,#3.8/.6,
        pre_samples_add=10,
        post_samples_add=25)
    ldax.merge_islands(d_find_mask, width=50)
    
    # calculate per-channel-per-event baseline curves from a running average, masked of pulses
    d_bs_est = ldax.baseline_update(d, d_find_mask, alpha=0.08)
    
    # subtract baselines
    d = d - d_bs_est
    
    # compute the full event areas (summed over channels)
    e_area_raw = d.sum(axis=1).sum(axis=-1)
    
    # perform low-pass filter to suppress HF noise:
    d_filt = ldax.lowpass_RC(d, 0.1, n=10)
    
    # find pods, creating a boolean per-channel array that identifies pods
    pod_bool = ldax.pod_boolean(d_filt, thresh=2.5, prepod_samples=15, postpod_samples=20)
    
    # suppress the baseline of each waveform outside of a pod:
    d_filt[~pod_bool] = 0.
    
    # calculate the sum waveforms (summed over channels)
    d_chsum = d_filt[:,:32,:].sum(axis=1)
    
    # recompute another podding, based on sum waveform
    d_chsum_pod = ldax.pod_boolean(d_chsum, thresh=3., prepod_samples=15, postpod_samples=20)
    ldax.merge_islands(d_chsum_pod, width=10)
    
    # Make sure the start and end of the waveform are not part of a pulse
    d_chsum_pod[:,0] = False
    d_chsum_pod[:,-1] = False
    
    # Get the pod starts and stops -- this will form the first guess at the pulse boundaries.
    p_starts_evt, p_starts_samp = np.nonzero(np.diff(d_chsum_pod.astype(np.int8),axis=-1)==1)
    p_stops_evt, p_stops_samp = np.nonzero(np.diff(d_chsum_pod.astype(np.int8),axis=-1)==-1)
    
    # check that there are the same number of starts and stops in each event
    if len(p_starts_evt) != len(p_stops_evt):
        raise ValueError("The number of pulse starts and stops must be the same in each event")
    if not (p_starts_evt == p_stops_evt).all():
        raise ValueError("The number of pod starts and stops is not the same")
    
    # calculate the boundaries of each pulse
    pulse_sarray = np.empty(d_chsum.shape[0], dtype=np.uint16)
    for k in range(d_chsum.shape[0]):
        n_pulses = (p_starts_evt==k).sum()
        pulse_sarray[k] = n_pulses
    p_bnds = va.varray(
        darray=np.vstack([p_starts_samp+1, p_stops_samp]), sarray=pulse_sarray, dtype=np.uint32)
    
    # Split off low-amplitude tails from pulses; needs to be done several times
    split_dict = {
        'amp_frac': 0.04,
        'amp_max': 40.,
        'quiet_samples': 60,
        'buffer_samples': 10}
    p_bnds = va.varray(ldax.split_pulses(p_bnds.flatten(), p_bnds.sarray,d_chsum, **split_dict))
    p_bnds = va.varray(ldax.split_pulses(p_bnds.flatten(), p_bnds.sarray,d_chsum, **split_dict))
    p_bnds = va.varray(ldax.split_pulses(p_bnds.flatten(), p_bnds.sarray,d_chsum, **split_dict))
    p_bnds = va.varray(ldax.split_pulses(p_bnds.flatten(), p_bnds.sarray,d_chsum, **split_dict))
    
    # calculate pulse areas (sum and individual)
    p_area = va.varray(darray=ldax.get_pA(d_chsum, p_bnds.flatten(), p_bnds.sarray), sarray=p_bnds.sarray)
    p_area_ch = va.varray(
        darray=ldax.get_pA_ch(d_filt[:,:32,:], p_bnds.flatten(), p_bnds.sarray), 
        sarray=p_bnds.sarray)
    
    # calculate pulse heights (sum and individual)
    p_height = va.varray(darray=ldax.get_pH(d_chsum, p_bnds.flatten(), p_bnds.sarray), sarray=p_bnds.sarray)
    p_height_ch = va.varray(
        darray=ldax.get_pH_ch(d_filt[:,:32,:], p_bnds.flatten(), p_bnds.sarray), 
        sarray=p_bnds.sarray)
    
    # calculate pulse-area-fraction times
    area_fracs = np.r_[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    pA_list = [
        va.varray(darray=ldax.get_aft(d_chsum, p_bnds.flatten(), p_bnds.sarray, area_frac=ff), sarray=p_bnds.sarray)
        for ff in area_fracs]
    p_aft = va.inner_stack(pA_list, axis=1)
    del pA_list
    
    # calculate pulse widths
    p_width_9010 = p_aft[:,-2,:] - p_aft[:,1,:]
    p_width_7525 = p_aft[:,-3,:] - p_aft[:,2,:]
    
    # determin n-fold coincidence
    p_nfold = (p_area_ch > (1e-3)).sum(axis=1)
    
    # perform first-pass pulse classification
    # pulse identity: 0 (other); 1 (s1); 2 (s2); 3 (SPE)
    p_class = va.zeros(p_area.sarray, dtype=np.uint8)
    cut_S1 = (p_width_9010>10.) & (p_width_9010<20.) & \
        (p_width_7525 < (p_width_9010*.5+2.)) & (p_width_7525 > (p_width_9010*.37+.5))
    p_class[cut_S1 & (p_nfold>=2)] = 1
    p_class[cut_S1 & (p_nfold<2)] = 3
    cut_S2 = (p_width_9010>60.) & (p_width_9010<225.) & \
        (p_width_7525 < (p_width_9010*.75-7.)) & (p_width_7525 > ((p_width_9010**.55)*5.-25.))
    p_class[cut_S2] = 2
    
    # Move now from pulse-level quantities to identifying prominent S1s and S2s
    
    # construct cut for prominent S1 pulses
    S1A_max_ma = p_area[cut_S1].max(axis=-1)
    S1A_max = S1A_max_ma.data
    S1A_max_mask = S1A_max_ma.mask
    S1A_max[S1A_max_mask] = 0.
    pA_S1max = va.expand_to_columns(S1A_max, sarray=p_area.sarray)
    cut_S1_prominent = (p_class==1) & (p_area > 0.1*pA_S1max)
    
    # construct cut for prominent S2 pulses
    S2A_max_ma = p_area[cut_S2].max(axis=-1)
    S2A_max = S2A_max_ma.data
    S2A_max_mask = S2A_max_ma.mask
    S2A_max[S2A_max_mask] = 0.
    pA_S2max = va.expand_to_columns(S2A_max, sarray=p_area.sarray)
    cut_S2_prominent = (p_class==2) & (p_area > 0.05*pA_S2max)
    
    # create S1 RQs
    s1_area              = p_area[cut_S1_prominent]
    s1_area_ch           = p_area_ch[cut_S1_prominent]
    s1_height            = p_height[cut_S1_prominent]
    s1_height_ch         = p_height_ch[cut_S1_prominent]
    s1_pulse_bounds      = p_bnds[cut_S1_prominent]
    s1_aft               = p_aft[cut_S1_prominent]
    s1_width1090         = p_width_9010[cut_S1_prominent]
    
    # create S2 RQs
    s2_area              = p_area[cut_S2_prominent]
    s2_area_ch           = p_area_ch[cut_S2_prominent]
    s2_height            = p_height[cut_S2_prominent]
    s2_height_ch         = p_height_ch[cut_S2_prominent]
    s2_pulse_bounds      = p_bnds[cut_S1_prominent]
    s2_aft               = p_aft[cut_S2_prominent]
    s2_width1090         = p_width_9010[cut_S2_prominent]
    
    # calculate drift time
    s2_drift_time = s2_aft[:,1,:] - va.expand_to_columns(np.array(s1_aft[:,1,0]), sarray=s2_area.sarray)
    
    # compute xy reconstruction
    ch_pos = np.array([
        [1.5,1.5],
        [1.5,0.5],
        [0.5,0.5],
        [0.5,1.5],
        [1.5,-0.5],
        [1.5,-1.5],
        [0.5,-1.5],
        [0.5,-0.5],
        [-0.5,-0.5],
        [-0.5,-1.5],
        [-1.5,-1.5],
        [-1.5,-0.5],
        [-0.5,1.5],
        [-0.5,0.5],
        [-1.5,0.5],
        [-1.5,1.5]])
    s2_top = s2_area_ch[:,:16,:].sum(axis=1)
    s2_top[s2_top<=0.] = 1.
    va_ch_pos_x = va.varray(
        darray=np.tile(ch_pos[:,0][:,np.newaxis],(1,int(s2_area.sarray.sum()))), 
        sarray=s2_area.sarray)
    va_ch_pos_y = va.varray(
        darray=np.tile(ch_pos[:,1][:,np.newaxis],(1,int(s2_area.sarray.sum()))), 
        sarray=s2_area.sarray)
    s2_x_raw = (s2_area_ch[:,:16,...]*va_ch_pos_x).sum(axis=1) / s2_top
    s2_y_raw = (s2_area_ch[:,:16,...]*va_ch_pos_y).sum(axis=1) / s2_top
    
    # collect RQs into dictionary
    d = {}
    d['p_bnds'] = p_bnds
    d['p_area'] = p_area
    d['p_area_ch'] = p_area_ch
    d['p_height'] = p_height
    d['p_height_ch'] = p_height_ch
    d['area_fracs'] = area_fracs
    d['p_aft'] = p_aft
    d['p_width_1090'] = p_width_9010
    d['p_width_2575'] = p_width_7525
    d['p_nfold'] = p_nfold
    d['p_class'] = p_class
    d['p_cut_s1_prominent'] = cut_S1_prominent
    d['p_cut_s2_prominent'] = cut_S2_prominent
    d['s1_area'] = s1_area
    d['s1_area_ch'] = s1_area_ch
    d['s1_height'] = s1_height
    d['s1_height_ch'] = s1_height_ch
    d['s1_pulse_bounds'] = s1_pulse_bounds
    d['s1_aft'] = s1_aft
    d['s1_width1090'] = s1_width1090
    d['s2_area'] = s2_area
    d['s2_area_ch'] = s2_area_ch
    d['s2_height'] = s2_height
    d['s2_height_ch'] = s2_height_ch
    d['s2_pulse_bounds'] = s2_pulse_bounds
    d['s2_aft'] = s2_aft
    d['s2_width1090'] = s2_width1090
    d['s2_drift_time'] = s2_drift_time
    d['ch_pos'] = ch_pos
    d['s2_x_raw'] = s2_x_raw
    d['s2_y_raw'] = s2_y_raw
    
    return d

def main(argv):
    fName = argv[1]
    fName_list = fName.split('.')
    rqName_list = [item for item in fName_list if item not in ('bin','gz')]
    rqName = '.'.join(rqName_list) + '_RQ.vrz'
    
    # Read file header and get data-info
    _, d_info, header = ldax.Read_DDC40_fName(f'{raw_data_path}/{fName}', num_events=2)
    num_events = header['num_events_in_file']
    
    num_events_per_iteration = min(1000, num_events)
    
    d = process_portion(f'{raw_data_path}/{fName}', 0, num_events_per_iteration)
    i_end = num_events_per_iteration
    while (i_end < num_events):
        num_events_load = min(i_end+num_events_per_iteration, num_events) - i_end
        print(f"...load events {i_end} through {i_end+num_events_load}")
        d_new = process_portion(f'{raw_data_path}/{fName}', i_end, num_events_load)
        for key in d:
            if isinstance(d[key], va.varray):
                d[key] = va.row_concat([d[key], d_new[key]])
        i_end += num_events_per_iteration
    
    # collect header info into RQs dict
    for item in header:
        d['h_'+item] = header[item]
    
    # collect data info into RQs dict
    for item in d_info:
        d['d_'+item] = d_info[item]
    
    # save RQs to file
    va.save(f'{rq_path}/{rqName}', **d)

if __name__ == "__main__":
    tracemalloc.start()
    try:
        main(sys.argv)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        print(f"\nPeak memory usage: {peak / 1024 / 1024 / 1024:.2f} GB")
        tracemalloc.stop()


