import c_ldax_proc as clp

window_length = 1030
sample_size_bytes = dtype(int16).itemsize

dname = '/mnt/drive1/SiPM_data'
#fname = '9-3-2024_0133_sipm_dco_0ADCtrig_5baseline.dat'
fname = '9-3-2024_0121_sipm_dco_5ADCtrig_5baseline.dat'
evt_block_size = int(2e5)

# figure out the file size:
with open(f'{dname}/{fname}','rb') as ff:
    ff.seek(0,2)
    filesize_bytes = ff.tell()
filesize_samples = int(filesize_bytes / sample_size_bytes)
filesize_events  = int(filesize_samples / window_length)
print(f'File contains {filesize_events} events')

n_baseline = 200
num_read_operations = int(ceil(filesize_events / evt_block_size))
print(f'Number of file read iterations: {num_read_operations}')

start_position = 0
idx_late_window = 500

h_wMax_x = linspace(-4,9,100)
h_wMax_n = zeros(len(h_wMax_x)-1)


# n = _np.histogramdd(_np.array([x,y]).T, [x_xe, x_ye])[0]
# c_[x,y] does the same as np.array([x,y]).T

if False:
    wMax_arr = empty(filesize_events)
    wMin_arr = empty(filesize_events)
    print('Progress (' + ' '*num_read_operations + ')', end='\r')
    print('Progress (', end='', flush=True)
    for k0 in range(num_read_operations):
        print('*', end='', flush=True)
        end_position = min(start_position + evt_block_size * window_length, filesize_samples)
        d_raw = empty((int((end_position-start_position)/window_length), window_length))
        with open(f'{dname}/{fname}','rb') as ff:
            d_raw.ravel()[:] = fromfile(
                ff,
                dtype=int16,
                count=end_position-start_position,
                offset=start_position)[:]
        bs_ave = d_raw[:,:n_baseline].mean(axis=1)
        d_raw -= tile(c_[bs_ave],(1,window_length))
        d_filt = clp.avebox(d_raw, 5)
        d_filt = clp.avebox(d_filt, 25)
        d_filt = clp.avebox(d_filt, 51)
        
        wMax = d_filt[:,idx_late_window:].max(axis=1)
        wMin = d_filt[:,idx_late_window:].min(axis=1)
        wMax_arr[int(start_position/window_length):int(end_position/window_length)] = wMax
        wMin_arr[int(start_position/window_length):int(end_position/window_length)] = wMin
        
        cut_neg = wMin > 0.55*wMax-3.5
        
        #h_wMax_n += histogram(wMax[cut_neg], h_wMax_x)[0]
        h_wMax_n += histogram(wMax, h_wMax_x)[0]
        
        start_position = end_position
h_wMax_n = histogram(wMax_arr, h_wMax_x)[0]
print('\n', flush=True)
figure(81); clf()
lstairs(h_wMax_x, h_wMax_n, '-')
xlabel('filtered window max [adcc]')
ylabel('Counts')

figure(82); clf()
plot2d(wMin_arr, wMax_arr, [-8,5],[-4,10],100,100,flag_log=True)
xlabel('window minimum [adcc]')
ylabel('window maximum [adcc]')

def plot_evt(evt_num):
    start_position = evt_num * window_length
    end_position = start_position + window_length
    with open(f'{dname}/{fname}','rb') as ff:
        d_evt = fromfile(ff, dtype=int16, count=end_position-start_position, offset=start_position)
    bs_ave_evt = d_evt[:n_baseline].mean()
    d_evt = d_evt - bs_ave_evt
    d_evt_filt = clp.avebox(d_evt, 5)
    d_evt_filt = clp.avebox(d_evt_filt, 25)
    d_evt_filt = clp.avebox(d_evt_filt, 51)
    a = plot(d_evt, '-', color=r_[1,1,1]*.5,lw=3)
    b = plot(d_evt_filt, 'r-', lw=1)
    
    return a+b


"""
d_raw = empty((evt_block_size, window_length))
with open(f'{dname}/{fname}','rb') as ff:
    d_raw.ravel()[:] = fromfile(ff, dtype=int16, count=evt_block_size*window_length)[:]

bs_ave = d_raw[:,:n_baseline].mean(axis=1)
d_raw -= tile(c_[bs_ave],(1, window_length))

d_filt = clp.avebox(d_raw, 5)
d_filt = clp.avebox(d_filt, 25)
d_filt = clp.avebox(d_filt, 51)

figure(31, figsize=(10.28, 8.97)); clf()
evt_offset = 100
for k in range(12):
    subplot(6,2,k+1)
    plot(d_raw[k+evt_offset,:],'-', color=r_[1,1,1]*.5,lw=2)
    plot(d_filt[k+evt_offset,:],'r-',lw=1)

idx_late_window = 500

wMax = d_filt[:,idx_late_window:].max(axis=1)
wMin = d_filt[:,idx_late_window:].min(axis=1)

figure(32); clf()
#plot(wMax,wMin,'.',markersize=2)
plot2d(wMax,wMin,[-3,7],[-7,4],100,100,flag_log=True)
xlabel('filtered window max [adcc]')
ylabel('filtered window min [adcc]')

cut_neg = wMin > .55 * wMax - 3.5
"""

