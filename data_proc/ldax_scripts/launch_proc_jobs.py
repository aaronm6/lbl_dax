"""
The file 'ldax_proc_script.py' processes a single raw file, producing a single RQ file.

This script finds which raw files don't have a corresponding RQ file; it will start up to
4 raw-file-processings at a time.
"""

import os, subprocess
from time import sleep

raw_data_path = '/mnt/drive1/TPC_data'
rq_path = '/mnt/drive2/TPC_RQs'
#log_path = 

def file_tags(filename_list, extensions):
    if isinstance(extensions, str):
        extensions = (extensions,)
    file_tags = []
    for fname in filename_list:
        fname_list = fname.split('.')
        file_tag_list = [item for item in fname_list if item not in extensions]
        file_tags.append('.'.join(file_tag_list))
    return file_tags

def main():
    raw_files = os.listdir(raw_data_path)
    rq_files = os.listdir(rq_path)
    
    file_raw_tag = file_tags(raw_files, ('bin','gz'))
    file_rq_tags = file_tags(rq_files, ('vrz',))
    
    files_to_process = [item for item in file_raw_tag if item not in file_rq_tags]
    files_in_process = []
    files_processed = []
    
    files_in_process_dict = {}
    
    while files_to_process or files_in_process:
        while (len(files_in_process)<4) and files_to_process:
            file_proc = files_to_process.pop()
            print(f"--Processing {file_proc}", flush=True)
            popen_list = [
                'python',
                'ldax_proc_script.py',
                '-f',
                f'{file_proc}.bin.gz',
                '-c',
                '/home/aaronm/pylab/lbl_dax/data_proc/ldax_settings/proc_settings_v001.yaml']
            p = subprocess.Popen(popen_list)
            #p = subprocess.Popen(['python','ldax_proc_script.py',f'{file_proc}.bin.gz'])
            files_in_process.append(file_proc)
            files_in_process_dict[file_proc] = p
        # check for processes finished
        for file in files_in_process:
            if files_in_process_dict[file].poll() is not None:
                print(f"--finished {file}", flush=True)
                files_processed.append(file)
                files_in_process.remove(file)
                del files_in_process_dict[file]
        sleep(10)
    print('********* finished ******')

if __name__ == "__main__":
    main()

