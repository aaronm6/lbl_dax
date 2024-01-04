if 'os' not in locals():
    import os
if 'sys' not in locals():
    import sys

#if 'cpr' not in locals():
#    import c_processmodule as cpr

dirname = '/mnt/drive1/PMT_data/'
fnames = os.listdir(dirname)
itemsize = dtype('int16').itemsize
event_length = 520

file_idx = 2 # of the files in fnames list, load the element with this index

load_first = 0 # the number of the first event to load (0 being first)
num_load = 10 # load this many files

with open(f'{dirname}{fnames[file_idx]}', 'rb') as ff:
    ff.seek(0*itemsize,0)
    d = fromfile(ff, dtype=int16, count=num_load*event_length)
    d = d.reshape((num_load,event_length))

