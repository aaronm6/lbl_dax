if 'os' not in locals():
    import os
if 'sys' not in locals():
    import sys

if 'cpr' not in locals():
    import c_processmodule as cpr

dirname = '/mnt/drive1/PMT_data/'
fnames = os.listdir(dirname)
itemsize = dtype('int16').itemsize
event_length = 520
t_interval = 1e-8 # time, in seconds, between samples.  I THINK this is right

file_idx = 2 # of the files in fnames list, load the element with this index

load_first = 0 # the number of the first event to load (0 being first)
num_load = 10 # load this many files

if 'd' not in locals():
    with open(f'{dirname}/wave0_15ADC.dat', 'rb') as ff:
        ff.seek(0*itemsize,0)
        #d = fromfile(ff, dtype=int16, count=num_load*event_length).reshape([-1,event_length])
        d = fromfile(ff, dtype=int16).reshape([-1,event_length])

t_vec = r_[:event_length].astype(float) * t_interval
k_show = 1

# highpass(t, v, f_c)
fc_hp = 5e5
bl_1st100 = d[k_show,:100].astype(float).mean()
d_k_filt = cpr.highpass(t_vec, d[k_show,:].astype(float), fc_hp, first_el=bl_1st100)

figure(41, figsize=(10,8.5)); clf()

subplot(2,1,1)
plot(t_vec*1e6, d[k_show,:],'k-',lw=2)
plot(t_vec*1e6, d_k_filt + d[k_show,:100].mean() - d_k_filt[:100].mean(), 'r-', lw=0.75)
xlim([0,event_length*t_interval*1e6])
ylabel('raw signal')
grid()
title(f'k = {k_show}')
subplot(2,1,2)
plot(t_vec*1e6, d_k_filt, 'b-', lw=1)
xlim([0,event_length*t_interval*1e6])
xlabel('time [$\mu$s]')
ylabel('high-passed signal')
grid()




