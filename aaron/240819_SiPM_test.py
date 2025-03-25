import c_ldax_proc as clp

def get_filesize(path_and_filename):
    """
    Returns the size of the provided file, in number of bytes.
    """
    with open(path_and_filename) as ff:
        ff.seek(0,2)
        filesize_bytes = ff.tell()
    return filesize_bytes

dirname = '/mnt/drive1/SiPM_data'
#fname = '8-19-2024_0228_ch0-sipm.dat'
#fname = '8-19-2024_0301_ch0-sipm.dat'
fname = '8-19-2024_0533_ch0-sipm.dat'
event_length = 1030
t_interval = 1e-8
itemsize = dtype('int16').itemsize
n_baseline = 200

if 'd_raw' not in locals():
    filesize_bytes = get_filesize(f'{dirname}/{fname}')
    num_events = int(filesize_bytes / itemsize / event_length)
    d_raw = empty((num_events, event_length))
    with open(f'{dirname}/{fname}', 'rb') as ff:
        d_raw.ravel()[:] = fromfile(ff, dtype=int16)[:]
        bs_ave = d_raw[:,:n_baseline].mean(axis=1)
        d_raw -= tile(c_[bs_ave],(1,event_length))

if 'd_filt' not in locals():
    d_filt = clp.avebox(d_raw, 5)
    #d_filt = clp.avebox(d_filt, 11)
    d_filt = clp.avebox(d_filt, 25)
    d_filt = clp.avebox(d_filt, 51)

#figure(21, figsize=(10.28,8.97)); clf()
figure(21, figsize=(9.56, 13.65)); clf()
evt_offset = 0
for k in range(12):
    subplot(6,2,k+1)
    if k==0:
        title(f'\\texttt{{{fname}}}')
    if k==10:
        xlabel('Time [samples]')
        ylabel('Voltage [adcc]')
    plot(d_raw[k+evt_offset,:],'-',color=r_[1,1,1]*.5,lw=2)
    plot(d_filt[k+evt_offset,:],'r-',lw=1)
    xlim([0, event_length])
    grid()
    ylim(ylim())

pH = d_filt.max(axis=1)
hH = S()
hH.x = linspace(0, 30, 80)
hH.n = histogram(pH, hH.x)[0]
figure(22); clf()
stairs(hH.x, hH.n, '-')
xlabel('Filtered Pulse Height [adcc]')
ylabel('Counts')
minorticks_on()

pA = d_filt.sum(axis=1)
hA = S()
hA.x = linspace(0,pA.max()*1.1,70)
hA.n = histogram(pA, hA.x)[0]
figure(23); clf()
stairs(hA.x, hA.n, '-')
xlabel('Filtered pulse area [adcc$\\times$samples]')
ylabel('Counts')

# Derive template pulse shape at fixed position, normalize to unity
f_i = d_filt.sum(axis=0)
#f_i = d_raw.sum(axis=0)
f_i /= f_i.max()
#f_i[:310] = 0.

f_i_mat = tile(f_i, (d_filt.shape[0],1))
H = (d_filt*f_i_mat).sum(axis=1) / ((f_i**2).sum())
figure(24); clf()
#plot(H,'.')
pH_tmpl = S()
pH_tmpl.x = linspace(0,20,100)
#pH_tmpl.x = linspace(20,50,75)
pH_tmpl.n = histogram(H, pH_tmpl.x)[0]
stairs(pH_tmpl.x, pH_tmpl.n,'-')
xlabel('Template pulse height [adcc]')
ylabel('Counts')

figure(25); clf()
plot(d_raw.sum(axis=0),'-')
plot(d_filt.sum(axis=0),'-')
xlabel('Time [samples]')
ylabel('Sum signal')
leg([0,1], 'Raw signals', 'Filtered signals', fontsize=10)

figure(21)
for k in range(12):
    subplot(6,2,k+1)
    plot(H[k+evt_offset]*f_i,'-', color=r_[0,1,0], lw=1)

figure(26); clf()
pA_230_285 = d_raw[:,230:285].sum(axis=1)
pA_x = linspace(-pA_230_285.max()*.25, pA_230_285.max()*1.1, 100)
pA_n = histogram(pA_230_285, pA_x)[0]
stairs(pA_x, pA_n,'-')
xlabel('Raw area in [230,285] samples [adc samples]')
ylabel('Counts')

figure(27); clf()
pH_230_285 = d_filt[:,230:285].max(axis=1)
pH_x = linspace(-pH_230_285.max()*.25,pH_230_285.max()*1.1, 100)
pH_n = histogram(pH_230_285, pH_x)[0]
stairs(pH_x, pH_n, '-')
xlabel('Filtered height in [230,285] samples [filtered adcc]')
ylabel('Counts')

figure(28); clf()
#plot(pH, pH_230_285, '.',markersize=2)
plot2d(pH, pH_230_285, [9.8,22], [-1.5,2.5], 75, 75)
xlabel('Pulse height')
ylabel('Pulse height in [230,285]')
colorbar()
cbarylabel('Counts')