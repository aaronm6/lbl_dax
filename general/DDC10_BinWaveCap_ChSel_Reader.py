# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:43:51 2018

@author: ErykD
"""

import numpy as np
import matplotlib.pyplot as plt

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
            waveArr[ich,ievt,:] = waveTmp
            dummy = np.fromfile(fp,dtype=np.uint32,count=1)
    fp.close()
    return (waveArr,waveInfo)

#%%
waveArr,waveInfo = Read_DDC10_BinWaveCap_ChSel('/home/dreamer/tmpData/EMS/TestCap_0x3FF_8192_1000_v12.bin')

#%%

plt.figure()
for i in range(10):
    for ich in range(10):
        plt.plot(waveArr[ich,i,:])
        #plt.plot(waveArr[1,i,:])
#plt.xlim((1000,2000))
plt.show()