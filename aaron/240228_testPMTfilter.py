import os, sys
import c_ldax_proc as clp

dirname = os.path.expanduser('~/data/HydroX/')
fName = 'wave0_15ADC.dat'
#fName = 'wave0_10ADC.dat'
event_length = 520
t_interval = 1e-8 # time, in seconds, between samples.
itemsize = dtype('int16').itemsize

load_filt = True
if 'd_raw' in locals():
    load_filt = False

# Load the raw data and feed into data structure: 2d array (each row is an event)
if load_filt:
    #with open(f'{dirname}/wave0_10ADC.dat', 'rb') as ff:
    with open(f'{dirname}/{fName}', 'rb') as ff:
        #ff.seek(0,0)
        ff.seek(0,2) # go 0 bytes away from the end of the file stream
        filesize_bytes = ff.tell()
        ff.seek(0,0)
        num_events = int(filesize_bytes / itemsize / event_length)
        d_raw = empty((num_events, event_length))
        d_raw.ravel()[:] = fromfile(ff, dtype=int16)[:]
        # Calculate baseline and subtract
        n_baseline = 100 # number of samples over which to average
        bs_s = d_raw[:,:n_baseline].mean(axis=1) # baseline from the start of the trace
        bs_e = d_raw[:,(-n_baseline):].mean(axis=1) # baseline from the end of the trace
        bs_ave = .5 * (bs_s + bs_e)
        
        d_raw = -d_raw + tile(c_[bs_ave],(1,event_length))


if load_filt:
    with tictoc():
        d_filt = clp.avebox(d_raw, 5)
        d_filt = clp.avebox(d_filt, 11)

rcParams.update({'font.size':10.})

bnds_trigPulse = r_[165, 210]
#bnds_off = r_[300,300+diff(bnds_trigPulse)]
bnds_off = r_[110,110+diff(bnds_trigPulse)]

figure(51, figsize=(7,10)); clf()
evt_offset = int(1e5) +17
for k in range(6):
    subplot(6,1,k+1)
    if k==0:
        title(f'{fName}')
    plot(d_raw[k+evt_offset,:],'-',color=r_[1,1,1]*.5,lw=2)
    plot(d_filt[k+evt_offset,:],'r-',lw=1)
    xlim([0, event_length])
    grid()
    ylim(ylim())
    #plot(r_[r_[1,1]*bnds_trigPulse[0],nan,r_[1,1]*bnds_trigPulse[1]],r_[ylim(),nan,ylim()],'b--',lw=1)
    fill(r_[bnds_trigPulse[0], bnds_trigPulse[1],bnds_trigPulse[1], bnds_trigPulse[0]],
        r_[ylim()[0],ylim()[0],ylim()[1],ylim()[1]],facecolor='b',edgecolor='none',alpha=.2)
    #plot(r_[r_[1,1]*bnds_off[0],nan,r_[1,1]*bnds_off[1]],r_[ylim(),nan,ylim()],'c--',lw=1)
    fill(r_[bnds_off[0],bnds_off[1],bnds_off[1],bnds_off[0]],
        r_[ylim()[0],ylim()[0],ylim()[1],ylim()[1]],facecolor='c',edgecolor='none',alpha=.2)
xlabel('Time [samples]')
ylabel('Voltage [adcc]')

pA_trig = d_filt[:,bnds_trigPulse[0]:bnds_trigPulse[1]].sum(axis=1)
#pA_trig = d_raw[:,bnds_trigPulse[0]:bnds_trigPulse[1]].sum(axis=1)
pH_trig = d_filt[:,bnds_trigPulse[0]:bnds_trigPulse[1]].max(axis=1)
rcParams.update({'font.size':15.})
figure(52); clf()
#plot(pA_trig,'.',markersize=2)
plot2d(r_[:len(pA_trig)], pA_trig, [0,len(pA_trig)], [-20,300], 100,100,flag_log=True)
xlabel('Event number')
ylabel('Window trace area [bins * samples]')
title('Integral of trigger window')
grid()

pA_off = d_filt[:, bnds_off[0]:bnds_off[1]].sum(axis=1)
#pA_off = d_raw[:, bnds_off[0]:bnds_off[1]].sum(axis=1)
pH_off = d_filt[:, bnds_off[0]:bnds_off[1]].max(axis=1)
figure(53); clf()
#plot(pA_off,'.',markersize=2)
plot2d(r_[:len(pA_off)], pA_off, [0,len(pA_off)], [-20,300], 100,100)
xlabel('Event number')
ylabel('Window trace area [bins * samples]')
title('Integral away from trigger window')
grid()


# The shape of the on-trigger area spectrum changes, after about the 60-thousandth event.
h_start = int(6e4)
bnds_pA = r_[-50,300]
h_xA = linspace(*bnds_pA,100)
h_nA_trig = histogram(pA_trig[h_start:], h_xA)[0]
h_nA_off = histogram(pA_off[h_start:], h_xA)[0]

figure(54); clf()
lstairs(h_xA, h_nA_off, 'c-')
lstairs(h_xA, h_nA_trig,'b-')
xlim(bnds_pA)
#ylim([0, ylim()[1]])
xlabel('Pulse Area')
ylabel('Counts')
leg([1,0],"Trigger window","Away from trig window", fontsize=12)
grid()

figure(55); clf()
plot2d(r_[:len(pH_trig)], pH_trig,[0,len(pH_trig)],[0,30],100,100, flag_log=True)
xlabel('Event number')
ylabel('Pulse height')

h_xH = linspace(-1,30,100)
h_nH_trig = histogram(pH_trig[h_start:], h_xH)[0]
h_nH_off = histogram(pH_off[h_start:], h_xH)[0]

figure(56); clf()
lstairs(h_xH, h_nH_off,'c-')
lstairs(h_xH, h_nH_trig,'b-')
xlabel('Filtered Pulse Height [au]')
ylabel('Counts')
xlim([-1,30])
leg([1,0],"Trigger window","Away from trig window", fontsize=12)
grid()

figure(57); clf()
plot2d(pA_trig, pH_trig,[-20,300],[-2,30],100,100, flag_log=True)
xlabel('Pulse Area [au]')
ylabel('Filtered Pulse Height [au]')
colorbar()
cbarylabel('Counts')
title('Trigger Window')

figure(58); clf()
plot2d(pA_off, pH_off,[-20,300],[-2,30],100,100, flag_log=True)
xlabel('Pulse Area [au]')
ylabel('Filtered Pulse Height [au]')
colorbar()
cbarylabel('Counts')
title('Off-trigger')

