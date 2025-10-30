"""
ldax_processing: functions for processing raw data in ldax
Functions are listed below with brief descriptions.  Refer to individual docstrings
for more information and usage.

Functions for reading raw data:
 - Read_DDC10_fHandle : Reads a raw or gzipped file, given by the already opened file handle
 - Read_DDC10_fName   : Reads a raw or gzipped file, given by the file name

Functions for filtering:
 - lowpass_RC: An n-pole lowpass RC filter of a time series data signal.
 - lowpass_exp: An exponential lowpass filter of a time series data signal.
 - avebox: A rolling-average filter (i.e. a convolution with a box function)
 - exp_filt: A convolution of a decaying-exponential
 - crosscorr: Perform a cross-correlation filter of a waveform with a given kernel, aka
              a template function that represents the pulse shape that is being searched
              for.

Processing function:
 - find_peaks: Takes a time-series waveform, looks for peaks and calculates peak
               properties like pulse area, pulse height, timing information, etc.
               Returns a dict with per-event information.

"""
from ldax_processing.c_ldax_proc import *
from ldax_processing.fft_filters import *
from ldax_processing.DDC10_BinReader import Read_DDC10_fName
from ldax_processing.DDC40_BinReader import Read_DDC40_fName, Read_DDC40_Header

del c_ldax_proc
del fft_filters
del DDC10_BinReader
