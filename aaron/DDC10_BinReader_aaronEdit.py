# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:43:51 2018

@author: ErykD

Mar 25 14:09:54 2025
Modified to clean up reading code to be safer via context block, stripped non-reading functionality
@author: AaronM
"""

import numpy as np
from scipy import signal

def bitfield(n):
    tmp = [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    return tmp[::-1]

def Read_DDC10_fHandle(fp):
    waveInfo = {}
    
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
    
    waveArr = np.empty((numChannels,numEvents,numSamples),dtype=np.int16)
    
    for ievt in range(numEvents):
        for ich in range(numChannels):
            _ = np.fromfile(fp,dtype=np.uint32,count=2)
            waveTmp = np.fromfile(fp,dtype=np.int16,count=numSamples)
            if waveTmp.size:
                waveArr[ich,ievt,:] = waveTmp
            _ = np.fromfile(fp,dtype=np.uint32,count=1)
    return waveArr, waveInfo

def Read_DDC10_fName(fName):
    with open(fName,'rb') as ff:
        waveArr, waveInfo = Read_DDC10_fHandle(ff)
    return waveArr, waveInfo
