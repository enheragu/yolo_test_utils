import os
import tarfile
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import shutil

from log_utils import log, bcolors

folder_compress = 'labels'

def get_folder_info(folder_path):
    total_size = 0
    total_files = 0
    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)
        for filename in files:
            file_path = os.path.join(root, filename)
            total_size += os.path.getsize(file_path)
    return total_size, total_files

def create_tar(dict_info, check_path):
    label_path = dict_info['label']
    output_path = dict_info['output']
    total_size, total_files = dict_info['info']
    members = [os.path.join(label_path, file) for file in os.listdir(label_path)]
    
    with tarfile.open(output_path, mode="w:gz") as tar:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=str(output_path).replace(check_path, "")) as pbar:
            processed_files = 0
            for member in members:
                arcname = 'labels/' + os.path.basename(member)
                tar.add(member, arcname=arcname)
                processed_files += 1
                pbar.update(os.path.getsize(member))
                pbar.set_postfix(files=f"{processed_files}/{total_files}")
    shutil.rmtree(label_path)

def compress_output_labels(check_path, folder_compress_name = folder_compress):
    log(f"Compressing '{folder_compress_name}' from {check_path} path")

    to_be_compressed = []
    for root, dirs, files in os.walk(check_path):
        if folder_compress_name in dirs:
            label_path = os.path.join(root, folder_compress_name)
            output_path = os.path.join(root, f'{folder_compress_name}.tar.gz')
            info = get_folder_info(label_path)
            to_be_compressed.append({'label': label_path, 'output': output_path, 'info': info})
    
    # log(f"Files to be compressed are: {to_be_compressed}")
    with Pool() as pool:
        partial_create_tar = partial(create_tar, check_path=check_path)
        pool.map(partial_create_tar, to_be_compressed)
    
    log(f"Compressing finished")
        

if __name__ == '__main__':
    from config_utils import yolo_output_path
    folder_check = yolo_output_path # '/home/arvc/Desktop/detect_3080_copy/variance_day_visible_b5_kaist_trained/day_visible' #'/home/arvc/Desktop/detect_3080_copy'
    compress_output_labels(folder_check)


"""

--- Pre ---
[2024-02-29 10:06:40.641] Compressing 'labels' from /home/arvc/eeha/yolo_test_utils/runs/detect path
196G runs/
194G detect/

4414109 files txt

--- Post ---
[2024-02-29 10:53:44.971] Compressing finished
83G runs/
81G detect/

12  files txt
183 tar.gz generated

"""