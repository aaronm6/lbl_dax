# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:43:51 2018

@author: ErykD

Mar 25 14:09:54 2025
Modified to clean up reading code to be safer via context block, stripped non-reading functionality
@author: AaronM

May 22 10:06:30 2025
Simplified logic for function bitfield
Updated to have functionality to read gzipped files.
@author: AaronM
"""

import numpy as np
import gzip

def bitfield(n):
    return [int(digit) for digit in bin(n)[2:]][::-1]

def gz_fromfile(fHandle, dtype=float, count=-1, offset=-1):
    """
    A drop-in replacement for numpy.fromfile when reading from a binary file and the
    file is given as a file handle.  Here 'fHandle' is a file handle produced by
    gzip.open (although a regular python file handle should work as well).
    
    This is needed because numpy.fromfile does not work with gzip file handles.
    
    offset:
        if -1 (default), then stay at the current position in the file
        if >=0, then start at that offset FROM THE BEGINNING OF THE FILE (i.e. absolute offset)
    """
    # Set the start position of the read op
    # In np.fromfile, offset is the offset IN BYTES (not elements), so 
    # it naturally maps to offset
    if offset >= 0:
        fHandle.seek(offset, whence=0)
    
    # Read the specified data into a bytes buffer
    # In np.fromfile, size is the number of ITEMS of dtype, NOT bytes
    dtype_size = np.dtype(dtype).itemsize
    data_buff = fHandle.read(size=count*dtype_size)
    return np.frombuffer(data_buff, dtype=dtype)

def is_gzipped(fName):
    """
    Checks if file given by fName is a gzipped file
    Returns True if it is, False if it ain't
    """
    gzip_status = True
    with gzip.open(fName, 'r') as fh:
        try:
            fh.read(1)
        except gzip.BadGzipFile:
            gzip_status = False
    return gzip_status

def Read_DDC10_fHandle(fp):
    freader = gz_fromfile if isinstance(fp, gzip.GzipFile) else np.fromfile
    waveInfo = {}
    
    numEvents, numSamples, chSelMask, byteOrderPatternCode = \
        freader(fp, dtype=np.uint32, count=4)
    chMap = np.where(bitfield(chSelMask))[0]
    numChannels = len(chMap)
    byteOrderPattern = hex(byteOrderPatternCode)
    waveInfo['numEvents']   = numEvents
    waveInfo['numSamples']  = numSamples
    waveInfo['chMap']       = chMap
    waveInfo['numChannels'] = numChannels
    waveArr = np.empty((numChannels,numEvents,numSamples),dtype=np.int16)
    
    for ievt in range(numEvents):
        for ich in range(numChannels):
            _ = freader(fp,dtype=np.uint32,count=2)
            waveTmp = freader(fp,dtype=np.int16,count=numSamples)
            if waveTmp.size:
                waveArr[ich,ievt,:] = waveTmp
            _ = freader(fp,dtype=np.uint32,count=1)
    return waveArr, waveInfo

def Read_DDC10_fName(fName):
    """
    Read waveform data and metadata from a binary file from the DDC10.
    File given by input 'fName' must be in the format produced by the DDC10.  It CAN be gzipped.
    
    Outputs: 
        waveArr: Waveform data.  Format: 3D numpy array of dype np.int16. The dimensions are:
            (numChannels x numEvents x numSamples)
            where numSamples is the number of samples collected in a single event.
       waveInfo: Meta data in the form of a dict object
    """
    gzipStatus = is_gzipped(fName)
    openner = gzip.open if gzipStatus else open
    
    with openner(fName,'rb') as ff:
        waveArr, waveInfo = Read_DDC10_fHandle(ff)
    return waveArr, waveInfo
