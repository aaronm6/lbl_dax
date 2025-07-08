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

import os
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
    with gzip.open(fName, 'r') as fh:
        try:
            fh.read(1)
        except gzip.BadGzipFile:
            return False
    return True

def Read_DDC10_metadata(fp):
    """
    Read the metadata from the file
    """
    # Meta data is strictly at the beginning of the file, so we need to seek(0)
    fp.seek(0,0)
    
    # Select a 'fromfile' method that is appropriate for the type of file (raw or gzipped)
    freader = gz_fromfile if isinstance(fp, gzip.GzipFile) else np.fromfile
    
    # Initialize the metadata dict and fill it
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
    return waveInfo

def Read_DDC10_fHandle(fp, start_event=0, num_events=-1):
    """
    num_events=-1 means all remaining events in the file
    """
    
    # Select a 'fromfile' method that is appropriate for the type of file (raw or gzipped)
    freader = gz_fromfile if isinstance(fp, gzip.GzipFile) else np.fromfile
    
    # Get file size:
    filesize_bytes = fp.seek(0,2)
    fp.seek(0,0)
    
    # Read the metadata
    waveInfo = Read_DDC10_metadata(fp)
    
    # File position should now be advanced past the header, which is 16 bytes
    event_size_bytes = \
        waveInfo['numChannels'] * \
        (2 * np.dtype(np.uint32).itemsize + \
        waveInfo['numSamples']*np.dtype(np.int16).itemsize + \
        1 * np.dtype(np.uint32).itemsize)
    
    # determine how many events to read and make sure we're not trying to read
    # more events than are in the file
    num_evts_read = waveInfo['numEvents'] if num_events==-1 else num_events
    num_evts_read = min(num_evts_read, (waveInfo['numEvents']-start_event))
    
    waveInfo['numEventsRead'] = num_evts_read
    
    # Move to the position of the first event that you want to read
    fp.seek(start_event*event_size_bytes,1)
    
    # Initialize the array that holds the waveform data
    waveArr = np.empty((waveInfo['numChannels'],num_evts_read,waveInfo['numSamples']),dtype=np.int16)
    for ievt in range(num_evts_read):
        for ich in range(waveInfo['numChannels']):
            _ = freader(fp,dtype=np.uint32,count=2)
            waveTmp = freader(fp,dtype=np.int16,count=waveInfo['numSamples'])
            if waveTmp.size:
                waveArr[ich,ievt,:] = waveTmp
            _ = freader(fp,dtype=np.uint32,count=1)
    return waveArr, waveInfo

def Read_DDC10_fName(fName, start_event=0, num_events=-1):
    """
    Read waveform data and metadata from a binary file from the DDC10.
    File given by input 'fName' must be in the format produced by the DDC10.  It CAN be gzipped.
    
    Inputs:
          fName: Filename (including path, either absolute or relative) to read
    start_event: The first event to read.  Default is zero (i.e. start at the beginning)
     num_events: How many events to read.  The number of events available is the total
                 number of events in the file, minus start_event.  If more than this number
                 are requested, no warnings or errors are given, and the maximum events 
                 available will be read.  Default: -1 (which means all available)
    Outputs: 
        waveArr: Waveform data.  Format: 3D numpy array of dype np.int16. The dimensions are:
                 (numChannels x numEvents x numSamples)
                 where numSamples is the number of samples collected in a single event.
       waveInfo: Meta data in the form of a dict object
    """
    fName0 = os.path.expanduser(fName)
    gzipStatus = is_gzipped(fName0)
    openner = gzip.open if gzipStatus else open
    
    with openner(fName0,'rb') as ff:
        waveArr, waveInfo = Read_DDC10_fHandle(ff, start_event=start_event, num_events=num_events)
    waveInfo.update({'filename':fName0})
    return waveArr, waveInfo
