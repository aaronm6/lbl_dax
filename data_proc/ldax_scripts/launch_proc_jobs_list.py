"""
The file 'ldax_proc_script.py' processes a single raw file, producing a single RQ file.

This script finds which raw files don't have a corresponding RQ file; it will start up to
4 raw-file-processings at a time.
"""
import argparse
import os, subprocess
from time import sleep
import threading
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.console import Group

default_raw_data_path = '/mnt/drive1/TPC_data'
default_rq_path = '/mnt/drive2/TPC_RQs'

def parse_some_args():
    parser = argparse.ArgumentParser(description="Launch processing jobs of LDAX DDC40 data")
    parser.add_argument('-n', action='store', dest='num_procs', type=int, default=4,
        help="Number of concurrent processes to run")
    parser.add_argument('-r','--raw', action='store', dest='raw_data_path', 
        default=default_raw_data_path, help="The path in which to search for raw data")
    parser.add_argument('-p','--proc', action='store', dest='rq_path',
        default=default_rq_path, help="The path in which to search for and save RQ data")
    parser.add_argument('-f', action='store', dest='file_list',
        help="List of files to be processed")
    parser.add_argument('--conf', action='store', dest='conf_file',
        default='proc_settings_v001.yaml', help="Settings file for processing jobs")
    args = parser.parse_args()
    return args

def get_file_tags(filename_list, extensions=()):
    if isinstance(extensions, str):
        extensions = (extensions,)
    file_tags = []
    for fname in filename_list:
        fname_list = fname.split('.')
        file_tag_list = [item for item in fname_list if item not in extensions]
        file_tag_1 = '.'.join(file_tag_list)
        file_tag = file_tag_1.split('_RQ')[0]
        #file_tags.append('.'.join(file_tag_list))
        file_tags.append(file_tag)
    return file_tags

def run_job_wrapper(progress, task_id, popen_list):
    process = subprocess.Popen(
        popen_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1)
    
    for line in process.stdout:
        line = line.strip()
        if line.startswith('PROGRESS:'):
            parts = line.split(':', 1)[1].strip()
            percent_str = parts.split('%')[0].strip()
            percent = float(percent_str)
            description = parts.split('-',1)[1].strip() if '-' in parts else ""
            progress.update(task_id, completed=percent, description=description)
    process.wait()
    progress.remove_task(task_id)
    return process.returncode

def main():
    args = parse_some_args()
    raw_files = os.listdir(args.raw_data_path)
    rq_files = os.listdir(args.rq_path)
    files_to_process = []
    with open(args.file_list, 'r') as file:
        for line in file:
            files_to_process.append(line.strip())
    
    files_to_process = files_to_process[::-1]
    files_in_process = []
    files_processed = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[job_name]}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.description}"),
    ) as progress:
        while files_to_process or files_in_process:
            while (len(files_in_process)<args.num_procs) and files_to_process:
                file_proc = files_to_process.pop()
                #print(f"--Processing {file_proc}", flush=True)
                task_id = progress.add_task(file_proc, total=100, job_name=file_proc)
                popen_list = [
                    'python',
                    'ldax_proc_script.py',
                    '-f',
                    f'{file_proc}',
                    '-c',
                    args.conf_file]
                #p = subprocess.Popen(popen_list)
                # run_job_wrapper(progress, task_id, popen_list):
                thread = threading.Thread(
                    target=run_job_wrapper, 
                    args=(progress, task_id, popen_list,))
                thread.start()
                files_in_process.append(thread)
            # check for processes finished
            files_in_process = [item for item in files_in_process if item.is_alive()]
            sleep(1)
    print('********* finished ******')

if __name__ == "__main__":
    main()

