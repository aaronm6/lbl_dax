# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:43:51 2018

@author: ErykD
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
plt.rcParams.update({'font.size': 14})

def bitfield(n):
    tmp = [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    return tmp[::-1]


def Read_DDC10_BinWaveCap_ChSel(fName):
    #fName = "test.bin"
    waveInfo = {}
    
    fp = open(fName,"rb")
    
    numEvents = int(np.fromfile(fp,dtype=np.uint32,count=1))
    waveInfo['numEvents'] = numEvents
    numSamples = int(np.fromfile(fp,dtype=np.uint32,count=1))
    waveInfo['numSamples'] = numSamples
    chSelMask = int(np.fromfile(fp,dtype=np.uint32,count=1))
    chMap = np.where(bitfield(chSelMask))[0]
    waveInfo['chMap'] = chMap
    numChannels = len(chMap)
    waveInfo['numChannels'] = numChannels
    byteOrderPattern = hex(int(np.fromfile(fp,dtype=np.uint32,count=1)))
    
    waveArr = np.zeros((numChannels,numEvents,numSamples),dtype=np.int16)
    
    for ievt in range(numEvents):
        for ich in range(numChannels):
            dummy = np.fromfile(fp,dtype=np.uint32,count=2)
            waveTmp = np.fromfile(fp,dtype=np.int16,count=numSamples)
            if waveTmp.size:
            	waveArr[ich,ievt,:] = waveTmp
            dummy = np.fromfile(fp,dtype=np.uint32,count=1)
    fp.close()
    return (waveArr,waveInfo)

#%%
waveArr,waveInfo = Read_DDC10_BinWaveCap_ChSel('/var/nfs/general/data/241009/1620-12adcthreshold.bin')

#%%
print(waveInfo)
channel = 2
event = 5
plt.figure()
#for i in range(10):
print(waveArr.shape)
tempSignal = np.zeros(819)
for i in range(819):
    tempSignal[i] = np.sum(waveArr[0,2,10*i:10*(i+1)])/10

def boxcar_filter(data, window_size):
    kernel = np.ones(window_size) / window_size
    filtered_data = np.convolve(data, kernel, mode='same')
    
    # Handle edge effects
    half_window = window_size // 2
    filtered_data[:half_window] = filtered_data[half_window]
    filtered_data[-half_window:] = filtered_data[-half_window-1]
    
    return filtered_data

filter = signal.butter(10, 5e6, 'lp', fs=1e8, output='sos')
print(waveArr[0,2,:].shape)
filtered = signal.sosfilt(filter, waveArr[0,200,:])
#Sample implementation of recursive box filter
boxcarSignal = boxcar_filter(filtered, 10)
tempSignal = np.zeros(819)
for i in range(819):
    tempSignal[i] = np.sum(boxcarSignal[10*i:10*(i+1)])/10
#plt.plot(tempSignal)
plt.plot(filtered)
#for i in range(10):
#    for ich in range(10):
#        plt.plot(waveArr[ich,i,:])
        #plt.plot(waveArr[1,i,:])
#plt.xlim((1000,2000))
plt.show()
