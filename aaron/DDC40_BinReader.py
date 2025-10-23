import os
import numpy as np
import gzip
from tqdm import trange

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
    print(f'size requested: {count*dtype_size}')
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

def Read_DDC40_metadata(fp):
    """
    Read the header from the file
    """
    # Meta data is strictly at the beginning of the file, so we need to seek(0)
    fp.seek(0,0)
    
    # Select a 'fromfile' method that is appropriate for the type of file (raw or gzipped)
    freader = gz_fromfile if isinstance(fp, gzip.GzipFile) else np.fromfile
    
    # Initialize the metadata dict
    waveInfo = {}
    
    # Read metadata
    numEvents, numSamples = freader(fp, dtype=np.uint32, count=2)
    numChannels, = freader(fp, dtype=np.uint8, count=1)
    chMap = freader(fp, dtype=np.uint8, count=numChannels)
    
    # Fill metadata dict and return
    waveInfo['num_events_in_file'] = numEvents
    waveInfo['num_samples'] = numSamples
    waveInfo['num_channels'] = numChannels
    waveInfo['channel_map'] = chMap
    
    return waveInfo

def Read_DDC40_fHandle(fp, start_event=0, num_events=-1):
    """
    num_events=-1 means all remaining events in the file
    """
    # Select a 'fromfile' method that is appropriate for the type of file (raw or gzipped)
    freader = gz_fromfile if isinstance(fp, gzip.GzipFile) else np.fromfile
    
    # Get file size:
    filesize_bytes = fp.seek(0,2)
    fp.seek(0,0)
    
    # Read the metadata
    waveInfo = Read_DDC40_metadata(fp)
    
    # File position should now be advanced past the header (variable size)
    """
    event-level header:
        TrigTSArr:      uint64 * 1
        TrigSeqNumArr:  uint32 * 1
        ChHitVectorArr: uint64 * 1
    event-level data:
        waveforms:      int16 * num_samples * num_channels
    """
    event_size_bytes = \
        np.dtype(np.uint64).itemsize + np.dtype(np.uint32).itemsize + np.dtype(np.uint64).itemsize + \
        waveInfo['num_channels'] * waveInfo['num_samples'] * np.dtype(np.int16).itemsize
    
    # determine how many events to read and make sure we're not trying to read
    # more events than are in the file
    num_evts_read = waveInfo['num_events_in_file'] if num_events==-1 else num_events
    num_evts_read = min(num_evts_read, (waveInfo['num_events_in_file']-start_event))
    print(f'{num_evts_read = }')
    waveInfo['num_events_read'] = num_evts_read
    
    # Move to the position of the first event that you want to read
    print(f'{start_event = }')
    fp.seek(start_event*event_size_bytes,1)
    
    # Initialize the arrays that holds the waveform data and event-header data
    trig_timestamp = np.empty(num_evts_read, dtype=np.uint64)
    trig_seq_num = np.empty(num_evts_read, dtype=np.uint32)
    ch_hit_vector = np.empty(num_evts_read, dtype=np.uint64)
    waveforms = np.empty((num_evts_read, waveInfo['num_channels'], waveInfo['num_samples']), dtype=np.int16)
    
    # loop through events and fill the arrays
    #for k in trange(num_evts_read, desc="Reading file", leave=False):
    print('Entering event loop')
    for k in range(num_evts_read):
        print(f'{k = }')
        trig_timestamp[k], = freader(fp, dtype=np.uint64)
        trig_seq_num[k], = freader(fp, dtype=np.uint32)
        ch_hit_vector[k], = freader(fp, dtype=np.uint64)
        
        waveforms[k,...] = freader(
            fq, 
            dtype=np.int16, 
            count=waveInfo['num_channels']*waveInfo['num_samples']
        ).reshape(waveInfo['num_channels'],waveinfo['num_samples'])
    
    waveInfo['trig_timestamp'] = trig_timestamp
    waveInfo['trig_seq_num'] = trig_seq_num
    waveInfo['ch_hit_vector'] = ch_hit_vector
    
    return waveforms, waveInfo

def Read_DDC40_fName(fName, start_event=0, num_events=-1):
    """
    Read waveform data and metadata from a binary file from the DDC40.
    File given by input 'fName' must be in the format produced by the DDC40.  It CAN be gzipped.
    
    Inputs:
          fName: Filename (including path, either absolute or relative) to read
    start_event: The first event to read.  Default is zero (i.e. start at the beginning)
     num_events: How many events to read.  The number of events available is the total
                 number of events in the file, minus start_event.  If more than this number
                 are requested, no warnings or errors are given, and the maximum events 
                 available will be read.  Default: -1 (which means all available)
    Outputs: 
      waveforms: Waveform data.  Format: 3D numpy array of dype np.int16. The dimensions are:
                 (num_events x num_channels x num_samples)
                 where numSamples is the number of samples collected in a single event, single channel.
       waveInfo: Meta data in the form of a dict object, includes timestamps and sequence numbers.
    """
    fName0 = os.path.expanduser(fName)
    gzipStatus = is_gzipped(fName0)
    openner = gzip.open if gzipStatus else open
    
    with openner(fName0, 'rb') as ff:
        waveform, waveInfo = Read_DDC40_fHandle(ff, start_event=start_event, num_events=num_events)
    waveInfo.update({'filename':fName})
    return waveform, waveInfo








