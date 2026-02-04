import os, sys
import numpy as np
import ldax_processing as ldax
import ldax_analysis as lan
import varray as va
import tracemalloc
import argparse
import yaml
import re

def parse_some_args():
    parser = argparse.ArgumentParser(description="Process LDAX DDC40 data")
    parser.add_argument('-f', action='store', dest='raw_file', type=str, help="Name of raw file to process")
    parser.add_argument('-c','--conf', action='store', dest='conf_file', type=str,
        help="Name of YAML configuration file to use")
    parser.add_argument('-o','--output', action='store', dest='out_file', default='default',
        type=str, help="(optional) select the name of the RQ file")
    args = parser.parse_args()
    return args

class dict_attr(dict):
    """
    I'd like a dict whose items can be accessed like attributes
    e.g.:  d['a'] can be done as d.a
    """
    def __getattr__(self, item):
        return self.__getitem__(item)

def get_filename_ints_from_fullpath(filename_and_path):
    """
    Raw filenames might be e.g. '2026-02-02-1438.bin.gz' or '2026-02-03-0810_000003.bin.gz'
    We want to extract two integers from these filenames that make it easy to include them
    in arrays that can be carried with data sets when multiple datasets are compiled 
    together.  This function converts the filename into two integers:
        filename tag (uint64), filename iteration (int16)
    '2026-02-03-0810_000003.bin.gz' -> (202602030810, 3)
    '2026-02-02-1438.bin.gz'        -> (202602021438, -1)
    """
    file_no_suffix = re.search(f'(.*).bin',os.path.basename(filename_and_path)).groups()[0]
    tag_patt = r'(\d{4})-(\d{2})-(\d{2})-(\d{4})'
    iter_patt = r'_(\d+)'
    tag_str = ''.join(re.search(tag_patt, file_no_suffix).groups())
    tag_int = np.uint64(tag_str)
    iter_search = re.search(iter_patt, file_no_suffix)
    iter_int = np.int16(-1)
    if iter_search:
        iter_int = np.int16(iter_search.groups()[0])
    return tag_int, iter_int

def process_portion(filename_and_path, start_event, num_events, c):
    """
    c is a dict_attr object with loaded settings from the conf file
    """
    # process filename into ints
    fname_int, fname_iter = get_filename_ints_from_fullpath(filename_and_path)
    
    # load data
    d, _, _ = ldax.Read_DDC40_fName(filename_and_path, start_event=start_event, num_events=num_events)
    
    # determine number of events; should be the same as num_events, but this will see what it really is
    num_events_loaded = d.shape[0]
    
    # prepare the arrays that hold the event_id, the file_tag and the file_iteration
    event_id = np.r_[:num_events_loaded] + start_event
    file_tags = np.full(num_events_loaded, fname_int)
    file_iters = np.full(num_events_loaded, fname_iter)
    
    # convert waveform from dtype=int16 to dtype=float64
    d = d.astype(float)
    
    # determine baseline
    baseline_estimator = getattr(lan, c.baseline_estimator)
    d_bs_est = baseline_estimator(d, c)

    # subtract baselines
    d = d - d_bs_est
    
    # compute the full event areas (summed over channels)
    #e_area_raw = d.sum(axis=1).sum(axis=-1)
    
    # perform low-pass filter to suppress HF noise:
    filter_function = getattr(lan, c.filter_function)
    d_filt = filter_function(d, c)
    
    # create a per-channel boolean that is true where there are pods
    pod_bool_ch_func = getattr(lan, c.pod_bool_ch_func)
    pod_bool = pod_bool_ch_func(d_filt, c)
    
    # suppress the baseline of each waveform outside of a pod:
    d_filt[~pod_bool] = 0.
    
    # calculate the sum waveforms (summed over channels)
    d_chsum = d_filt[:,:32,:].sum(axis=1)
    
    # recompute another podding, based on sum waveform
    pod_bool_sum_func = getattr(lan, c.pod_bool_sum_func)
    d_chsum_pod = pod_bool_sum_func(d_chsum, c)
    
    get_p_bnds = getattr(lan, c.p_bnds_func)
    p_bnds = get_p_bnds(d_chsum, d_chsum_pod, c)
    
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
    area_fracs = np.array(c.p_area_fracs)#np.r_[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    pA_list = [
        va.varray(darray=ldax.get_aft(d_chsum, p_bnds.flatten(), p_bnds.sarray, area_frac=ff), sarray=p_bnds.sarray)
        for ff in area_fracs]
    p_aft = va.inner_stack(pA_list, axis=1)
    del pA_list
    
    # calculate pulse widths
    p_width_9010 = p_aft[:,-2,:] - p_aft[:,1,:]
    p_width_7525 = p_aft[:,-3,:] - p_aft[:,2,:]
    
    # determin n-fold coincidence
    p_nfold = (p_height_ch > c.ch_podbool_thresh).sum(axis=1)
    
    # perform first-pass pulse classification
    # pulse identity: 0 (other); 1 (s1); 2 (s2); 3 (SPE)
    p_class = va.zeros(p_area.sarray, dtype=np.uint8)
    cut_S1 = (p_width_9010>c.s1_9010_min) & (p_width_9010<c.s1_9010_max) & \
        (p_width_7525 < (p_width_9010*c.s1_7525_max_m+c.s1_7525_max_b)) & \
        (p_width_7525 > (p_width_9010*c.s1_7525_min_m+c.s1_7525_min_b))
    p_class[cut_S1 & (p_nfold>=c.s1_nfold)] = 1
    p_class[cut_S1 & (p_nfold<c.s1_nfold)] = 3
    cut_S2 = (p_width_9010>c.s2_9010_min) & (p_width_9010<c.s2_9010_max) & \
        (p_width_7525 < (p_width_9010*c.s2_7525_max_m+c.s2_7525_max_b)) & \
        (p_width_7525 > ((p_width_9010**c.s2_7525_min_p)*c.s2_7525_min_a+c.s2_7525_min_b))
    p_class[cut_S2] = 2
    
    # Move now from pulse-level quantities to identifying prominent S1s and S2s
    
    # construct cut for prominent S1 pulses
    S1A_max_ma = p_area[cut_S1].max(axis=-1)
    S1A_max = S1A_max_ma.data
    S1A_max_mask = S1A_max_ma.mask
    S1A_max[S1A_max_mask] = 0.
    pA_S1max = va.expand_to_columns(S1A_max, sarray=p_area.sarray)
    cut_S1_prominent = (p_class==1) & (p_area > c.s1_prom_max_frac*pA_S1max)
    
    # construct cut for prominent S2 pulses
    S2A_max_ma = p_area[cut_S2].max(axis=-1)
    S2A_max = S2A_max_ma.data
    S2A_max_mask = S2A_max_ma.mask
    S2A_max[S2A_max_mask] = 0.
    pA_S2max = va.expand_to_columns(S2A_max, sarray=p_area.sarray)
    cut_S2_prominent = (p_class==2) & (p_area > c.s2_prom_max_frac*pA_S2max)
    
    # create S1 RQs
    s1_area              = p_area[cut_S1_prominent]
    s1_area_ch           = p_area_ch[cut_S1_prominent]
    s1_height            = p_height[cut_S1_prominent]
    s1_height_ch         = p_height_ch[cut_S1_prominent]
    s1_pulse_bounds      = p_bnds[cut_S1_prominent]
    s1_aft               = p_aft[cut_S1_prominent]
    s1_width1090         = p_width_9010[cut_S1_prominent]
    s1_maf               = s1_area_ch.max(axis=1) / s1_area  # max area fraction
    s1_mhf               = s1_height_ch.max(axis=1) / s1_height # max height fraction
    
    # create S2 RQs
    s2_area              = p_area[cut_S2_prominent]
    s2_area_ch           = p_area_ch[cut_S2_prominent]
    s2_height            = p_height[cut_S2_prominent]
    s2_height_ch         = p_height_ch[cut_S2_prominent]
    s2_pulse_bounds      = p_bnds[cut_S2_prominent]
    s2_aft               = p_aft[cut_S2_prominent]
    s2_width1090         = p_width_9010[cut_S2_prominent]
    
    # load SiPM gains and apply
    gains_dir, gains_file = os.path.split(c.sipm_spe_areas)
    if not gains_dir:
        gains_dir = os.path.normpath(os.path.join(__file__,'..','..','ldax_settings'))
    sphe_ch = np.loadtxt(os.path.join(gains_dir, gains_file))
    sphe_ch = sphe_ch[:,1] # the second column (column 1) is the spe areas in adcc*samples
    p_phe_ch_darray  = p_area_ch.flatten() / sphe_ch[:, np.newaxis]
    s1_phe_ch_darray = s1_area_ch.flatten() / sphe_ch[:, np.newaxis]
    s2_phe_ch_darray = s2_area_ch.flatten() / sphe_ch[:, np.newaxis]
    
    p_phe_ch = va.varray(darray=p_phe_ch_darray, sarray=p_area_ch.sarray)
    p_phe = p_phe_ch.sum(axis=1)
    s1_phe_ch = va.varray(darray=s1_phe_ch_darray, sarray=s1_area_ch.sarray)
    s1_phe = s1_phe_ch.sum(axis=1)
    s2_phe_ch = va.varray(darray=s2_phe_ch_darray, sarray=s2_area_ch.sarray)
    s2_phe = s2_phe_ch.sum(axis=1)
    
    # calculate drift time
    s2_drift_time = s2_aft[:,1,:] - va.expand_to_columns(np.array(s1_aft[:,1,0]), sarray=s2_area.sarray)
    
    # compute xy reconstruction
    sipm2x2_center_positions_mm = 7.85
    #   the position of each of the four 2x2 sipm packages on a board
    sipm2x2_centers_mm = np.array([
        [1, 1],
        [1, -1],
        [-1, -1],
        [-1, 1]]).astype(float) * sipm2x2_center_positions_mm
    #   within a 2x2 sipm package, the position of the lower-left corner of each sipm channel
    sipm_i_rel_corner_positions_mm = np.array([
        [.25,.25],
        [.25,-6.15-.25],
        [-6.15-.25, -6.15-.25],
        [-6.15-.25,.25]])
    #   within a 2x2 sipm package, the position of the center of each sipm channel
    sipm_i_rel_center_positions_mm = sipm_i_rel_corner_positions_mm + (np.r_[1,1]*6.15/2)[np.newaxis,:]
    ch_pos = np.empty((sipm2x2_centers_mm.shape[0]*sipm_i_rel_center_positions_mm.shape[0], 2), dtype=float)
    for k4 in range(sipm2x2_centers_mm.shape[0]):
        for k1 in range(sipm_i_rel_center_positions_mm.shape[0]):
            ch_pos[k4*sipm2x2_centers_mm.shape[0] + k1,:] = \
                sipm2x2_centers_mm[k4,:] + sipm_i_rel_center_positions_mm[k1,:]
    # here with raw pulse areas in units of phe
    s2_top = s2_phe_ch[:,:16,:].sum(axis=1)
    s2_top[s2_top<=0.] = 1.
    va_ch_pos_x = va.varray(
        darray=np.tile(ch_pos[:,0][:,np.newaxis],(1,int(s2_phe.sarray.sum()))), 
        sarray=s2_phe.sarray)
    va_ch_pos_y = va.varray(
        darray=np.tile(ch_pos[:,1][:,np.newaxis],(1,int(s2_phe.sarray.sum()))), 
        sarray=s2_phe.sarray)
    s2_x_raw = (s2_phe_ch[:,:16,...]*va_ch_pos_x).sum(axis=1) / s2_top
    s2_y_raw = (s2_phe_ch[:,:16,...]*va_ch_pos_y).sum(axis=1) / s2_top
    
    # Calculate variance of x and y, and covariance of x,y
    s2_var_x_raw = (s2_phe_ch[:,:16,...]*(va_ch_pos_x**2)).sum(axis=1)/S2_top - (s2_x_raw**2)
    s2_var_y_raw = (s2_phe_ch[:,:16,...]*(va_ch_pos_y**2)).sum(axis=1)/S2_top - (s2_y_raw**2)
    s2_var_xy_raw = (s2_phe_ch[:,:16,...]*va_ch_pos_x*va_ch_pos_y).sum(axis=1)/S2_top - s2_x_raw * s2_y_raw
    
    # collect RQs into dictionary
    # e_ means an event-level quantity
    # p_ means pulse-level quantity (agnostic of classification)
    # s1_ means quantities applying to pulses that are prominent S1s
    # s2_ means quantities applying to pulses that are prominent S2s
    # ss_ means single-scatter quantities (these are ndarrays)
    d_out = {}
    d_out['e_event_id'] = event_id
    d_out['e_file_tags'] = file_tags
    d_out['e_file_iters'] = file_iters
    d_out['p_bnds'] = p_bnds
    d_out['p_area'] = p_area
    d_out['p_area_ch'] = p_area_ch
    d_out['p_phe'] = p_phe
    d_out['p_phe_ch'] = p_phe_ch
    d_out['p_height'] = p_height
    d_out['p_height_ch'] = p_height_ch
    d_out['ref_area_fracs'] = area_fracs
    d_out['p_aft'] = p_aft
    d_out['p_width_1090'] = p_width_9010
    d_out['p_width_2575'] = p_width_7525
    d_out['p_nfold'] = p_nfold
    d_out['p_class'] = p_class
    d_out['p_cut_s1_prominent'] = cut_S1_prominent
    d_out['p_cut_s2_prominent'] = cut_S2_prominent
    d_out['s1_area'] = s1_area
    d_out['s1_area_ch'] = s1_area_ch
    d_out['s1_phe'] = s1_phe
    d_out['s1_phe_ch'] = s1_phe_ch
    d_out['s1_height'] = s1_height
    d_out['s1_height_ch'] = s1_height_ch
    d_out['s1_pulse_bounds'] = s1_pulse_bounds
    d_out['s1_aft'] = s1_aft
    d_out['s1_width1090'] = s1_width1090
    d_out['s1_maf'] = s1_maf
    d_out['s1_mhf'] = s1_mhf
    d_out['s2_area'] = s2_area
    d_out['s2_area_ch'] = s2_area_ch
    d_out['s2_phe'] = s2_phe
    d_out['s2_phe_ch'] = s2_phe_ch
    d_out['s2_height'] = s2_height
    d_out['s2_height_ch'] = s2_height_ch
    d_out['s2_pulse_bounds'] = s2_pulse_bounds
    d_out['s2_aft'] = s2_aft
    d_out['s2_width1090'] = s2_width1090
    d_out['s2_drift_time'] = s2_drift_time
    d_out['ref_ch_pos'] = ch_pos
    d_out['s2_x_raw'] = s2_x_raw
    d_out['s2_y_raw'] = s2_y_raw
    # for convenience, calculate lateral coordinates in r, theta
    d_out['s2_r_raw'] = np.sqrt((s2_x_raw**2) + (s2_y_raw**2))
    d_out['s2_theta_raw'] = np.arctan2(s2_y_raw, s2_x_raw)
    
    # Now identify single scatters and create numpy arrays for them
    d_out['num_s1'] = d_out['s1_phe'].sarray
    d_out['num_s2'] = d_out['s2_phe'].sarray
    cut_ss = (d_out['num_s1'] == 1) & (d_out['num_s2'] == 1)
    d_keys = list(d_out)
    for item in d_keys:
        if item.startswith(('s1_','s2_','e_')):
            d_out[f'ss_{item}'] = d_out[item][cut_ss].flatten()
    d_out['ss_evt_num'] = np.r_[:d.shape[0]][cut_ss]
    d_out['ss_full_evt_phe'] = d_out['p_phe'].sum(axis=-1)[cut_ss].data
    d_out['ss_bad_phe'] = d_out['ss_full_evt_phe'] - d_out['ss_s1_phe'] - d_out['ss_s2_phe']
    
    return d_out

def main():
    args = parse_some_args()
    fName = args.raw_file
    
    fName_list = fName.split('.')
    if args.out_file == 'default':
        rqName_list = [item for item in fName_list if item not in ('bin','gz')]
        rqName = '.'.join(rqName_list) + '_RQ.vrz'
    else:
        rqName = args.out_file
    
    conf_dir, conf_file = os.path.split(args.conf_file)
    # check if a directory was provided.  If not, grab the default conf directory
    if not conf_dir:
        conf_dir = os.path.normpath(os.path.join(__file__, '..', '..', 'ldax_settings'))
    
    # read conf file
    with open(os.path.join(conf_dir, conf_file),'r') as ff:
        c = dict_attr(yaml.safe_load(ff))
    
    # Read file header and get data-info
    _, d_info, header = ldax.Read_DDC40_fName(f'{c.raw_data_path}/{fName}', num_events=2)
    num_events = header['num_events_in_file']
    
    num_events_per_iteration = min(1000, num_events)
    d_list = []
    d_list.append(process_portion(f'{c.raw_data_path}/{fName}', 0, num_events_per_iteration, c))
    i_end = num_events_per_iteration
    while (i_end < num_events):
        num_events_load = min(i_end+num_events_per_iteration, num_events) - i_end
        #print(f"...load events {i_end} through {i_end+num_events_load}")
        print(f"PROGRESS: {100*i_end/num_events}% - {fName}", flush=True)
        #d_new = process_portion(f'{c.raw_data_path}/{fName}', i_end, num_events_load, c)
        d_list.append(process_portion(f'{c.raw_data_path}/{fName}', i_end, num_events_load, c))
        """
        for key in d:
            if isinstance(d[key], va.varray):
                d[key] = va.row_concat([d[key], d_new[key]])
        """
        i_end += num_events_per_iteration
    
    d = {}
    d_keys = list(d_list[0])
    for key in d_keys:
        if not key.startswith('ref_'):
            if isinstance(d_list[0][key], va.varray):
                d[key] = va.row_concat([item[key] for item in d_list])
            elif isinstance(d_list[0][key], np.ndarray):
                d[key] = np.concatenate([item[key] for item in d_list], axis=-1)
        else:
            d[key] = d_list[0][key] # channel-position map, area fractions, etc.
    
    # collect header info into RQs dict
    for item in header:
        d['h_'+item] = header[item]
    
    # collect data info into RQs dict
    #for item in d_info:
    #    d['d_'+item] = d_info[item]
    
    # add info regarding config file
    d['config_version'] = c.config_version
    d['sipm_spe_areas'] = c.sipm_spe_areas
    if hasattr(ldax, 'version'):
        d['ldax_processing_version'] = ldax.version
    else
        d['ldax_processing_version'] = 'none'
    if hasattr(lan, 'version'):
        d['ldax_analysis_version'] = lan.version
    else
        d['ldax_analysis_version'] = 'none'
    
    
    # save RQs to file
    va.save(f'{c.rq_path}/{rqName}', **d)

if __name__ == "__main__":
    tracemalloc.start()
    try:
        main()
    finally:
        current, peak = tracemalloc.get_traced_memory()
        print(f"\nPeak memory usage: {peak / (2**30):.2f} GB")
        tracemalloc.stop()


