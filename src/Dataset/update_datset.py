#!/usr/bin/env python3
# encoding: utf-8

"""
    Checks that dataset exists with all options needed. If not downloads Kaist, prepares YOLO annotated version 
    (see kaist_to_yolo_annotations.py) and extras (see rgb_thermal_mix.py)
"""

import os

from argparse import ArgumentParser
from pathlib import Path
import requests
import xtarfile as tarfile
from tqdm import tqdm
from clint.textui import progress
import shutil

if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    sys.path.append('./src/Dataset')

from utils import FileLock
from utils import log, bcolors, parseYaml, dumpYaml, getTimetagNow
from Dataset.constants import dataset_options_keys, dataset_keys, kaist_path, kaist_yolo_dataset_path, llvip_yolo_dataset_path, llvip_path, repo_path
from Dataset.constants import dataset_options, dataset_generated_cache
from Dataset.KAIST.kaist_to_yolo_annotations import kaistToYolo
from Dataset.LLVIP.llvip_to_yolo_annotations import llvipToYolo
from Dataset.rgb_thermal_mix import make_dataset

def getKaistData():
    filename="kaist-cvpr15.tar.gz"
    url="https://onedrive.live.com/download?cid=1570430EADF56512&resid=1570430EADF56512%21109419&authkey=AJcMP-7Yp86PWoE"
    download_path = f"{kaist_path}/../{filename}"

    # make sure that kaist path exists
    Path(kaist_path).mkdir(parents=True, exist_ok=True)

    log(f"[UpdateDataset::getKaistData] Getting Kaist dataset from source. Downloading it to {download_path}. (Note that it can take quite some time).")
    r = requests.get(url, stream=True)
    with open(download_path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
            if chunk:
                f.write(chunk)
                f.flush()


    log(f"[UpdateDataset::getKaistData] Extracting data from {download_path}. (Note that it can take up to 10-20 minutes).")
    # with tarfile.open(download_path, 'r') as archive:
    #     archive.extractall()

    with tarfile.open(download_path, 'r') as tar:
        # Go over each member
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            # Extract member
            tar.extract(member=member)

"""
    Returns true if generated dataset does not match the expected options
"""
def resetDatset(options, dataset_format, rgb_eq, thermal_eq, distortion_correct = True, relabeling = True):
    global dataset_generated_cache

    if os.path.exists(dataset_generated_cache):
        data = parseYaml(dataset_generated_cache)

        # Dataset format does not affect -> all formats are generated
        #if 'dataset_format'in data and data['dataset_format'] == dataset_format and \
        if  'rgb_eq' in data and data['rgb_eq'] == rgb_eq and \
            'thermal_eq' in data and data['thermal_eq'] == thermal_eq and \
            'distortion_correct' in data and data['distortion_correct'] == distortion_correct and\
            'relabeling' in data and data['relabeling'] == relabeling:
            log(f"Previous dataset generated as: format {data['dataset_format']}; rgb_eq: {data['rgb_eq']}; thermal_eq: {data['thermal_eq']}; distortion_correct: {data['distortion_correct']}; relabeling: {data['relabeling']}. No reset needed.")
            return False
        # Just for logging
        elif 'dataset_format' in data and 'rgb_eq' in data and 'thermal_eq' in data:
            log(f"Previous dataset generated as: format {data['dataset_format']}; rgb_eq: {data['rgb_eq']}; thermal_eq: {data['thermal_eq']}; distortion_correct: {data['distortion_correct']}; relabeling: {data['relabeling']}")

    return True
    
def dumpCacheFile(option, dataset_format, rgb_eq, thermal_eq, distortion_correct = True, relabeling = True):
    global dataset_generated_cache

    if os.path.exists(dataset_generated_cache):
        data = parseYaml(dataset_generated_cache)
    else:
        data = {'options': [], 'dataset_format': '-', 'rgb_eq': '-', 'thermal_eq': '-'}

    data['options'].append(option)
    data['dataset_format'] = dataset_format
    data['rgb_eq'] = rgb_eq
    data['thermal_eq'] = thermal_eq
    data['last_update'] = getTimetagNow()
    data['distortion_correct'] = distortion_correct
    data['relabeling'] = relabeling
    data['options'] = list(set(data['options']))
    
    dumpYaml(dataset_generated_cache, data, mode = "w+")

def checkDataset(options = [], dataset_format = 'kaist_coco', rgb_eq = 'none', thermal_eq = 'none',
                      distortion_correct = True, relabeling = True):

    if options is None:
        log(f"[UpdateDataset::checkDataset] No options provided, no checking Kaist dataset", bcolors.WARNING)
        return

    if 'kaist' in dataset_format:
        lock_path = os.path.join(kaist_yolo_dataset_path, '../.dataset_generation.lock')
        dataset_path = kaist_path
        dataset_yolo_path = kaist_yolo_dataset_path
    elif 'llvip' in dataset_format:
        lock_path = os.path.join(llvip_yolo_dataset_path, '../.dataset_generation.lock')
        dataset_path = llvip_path
        dataset_yolo_path = llvip_yolo_dataset_path
    elif 'coco' in dataset_format:
        log(f"[UpdateDataset::checkDataset] Original COCO dataset does not need generation.", bcolors.WARNING)
        return
    else:
        log(f"[ERROR] [UpdateDataset::checkDataset] No conversion known for dataset format provided.", bcolors.ERROR)
        exit()
    # Locks to avoid re-generation of dataset while other scheduler is generating it
    
    log(f'[UpdateDataset::checkDataset] Try to acquire lock in {lock_path}.')
    with FileLock(lock_path) as lock:
        log(f'[UpdateDataset::checkDataset] Lock acquired in {lock_path}.')
        # Ensure input is a list
        if type(options) is not type(list()):
            options = [options]

        # Check if kaist dataset is already in the system
        if 'kaist' in dataset_format:
            if not os.path.exists(f"{dataset_path}/images"):
                log(f"[UpdateDataset::checkDataset] Kaist dataset could not be found in {dataset_path}. Downloading it from scratch.")
                getKaistData()
            else:
                log(f"[UpdateDataset::checkDataset] Kaist dataset found in {dataset_path}, no need to re-download.")
        
        # make sure that kaist-yolo path exists

        if os.path.exists(dataset_yolo_path) and resetDatset(options, dataset_format, rgb_eq, thermal_eq, distortion_correct, relabeling):
            log(f'[UpdateDataset::checkDataset] Deleting previous dataset generated as options does not match current request.', bcolors.WARNING)
            shutil.rmtree(dataset_yolo_path)

        Path(dataset_yolo_path).mkdir(parents=True, exist_ok=True)

        # Check that YOLO version exists or create it
        setfolders = [ f.path for f in os.scandir(dataset_yolo_path) if f.is_dir() ]
        options_found = [ f.name for f in os.scandir(setfolders[0]) if f.is_dir() ] if setfolders else []
        
        # log(f"{dataset_yolo_path = }")
        # log(f"[UpdateDataset::checkDataset] {setfolders = };\n{options_found =}\n")

        if 'lwir' not in options_found and 'visible' not in options_found:
            log(f"[UpdateDataset::checkDataset] YOLO version dataset for {dataset_format} could not be found in {dataset_yolo_path}. Generating new labeling for both lwir and visible sets.")
            
            if 'kaist' in dataset_format:
                kaistToYolo(dataset_format=dataset_format, rgb_eq=rgb_eq, thermal_eq=thermal_eq, distortion_correct=distortion_correct, relabeling=relabeling)
            elif 'llvip' in dataset_format:
                llvipToYolo(dataset_format=dataset_format, rgb_eq=rgb_eq, thermal_eq=thermal_eq, distortion_correct=distortion_correct, relabeling=relabeling)
            # Update with new options
            dumpCacheFile('lwir', dataset_format, rgb_eq, thermal_eq, distortion_correct, relabeling)
            dumpCacheFile('visible', dataset_format, rgb_eq, thermal_eq, distortion_correct, relabeling)
            setfolders = [ f.path for f in os.scandir(dataset_yolo_path) if f.is_dir() ]
            options_found = [ f.name for f in os.scandir(setfolders[0]) if f.is_dir() ] if setfolders else []
        else:
            log(f"[UpdateDataset::checkDataset] YOLO dataset for {dataset_format} found in {dataset_yolo_path}, no need to re-label it.")
        
        # log(f"{dataset_yolo_path = }")
        # log(f"[UpdateDataset::checkDataset] {setfolders = };\n{options_found =}\n")


        # Check that needed versions exist or create them
        for option in options:
            if option not in options_found:
                log(f"[UpdateDataset::checkDataset] Custom dataset for option {option} requested but not found in dataset folders. Generating it.")
                if "preprocess" in dataset_options[option]:
                    dataset_options[option]["preprocess"](dataset_format)
                make_dataset(option=option, dataset_format=dataset_format, rgb_eq=rgb_eq, thermal_eq=thermal_eq, yolo_version_dataset_path=dataset_yolo_path)
                dumpCacheFile(option, dataset_format, rgb_eq, thermal_eq, distortion_correct, relabeling)
            else:
                log(f"[UpdateDataset::checkDataset] Custom dataset for option {option} requested is already in dataset folder.")

if __name__ == '__main__':
    parser = ArgumentParser(description="Checks if dataset exists in expected location and generates it if not (download, extract, reformat or regenerate).")
    parser.add_argument('-o', '--option', action='store', dest='olist',
                        type=str, nargs='*', default=dataset_options_keys,
                        help=f"Extra options of the datasets to be included (apart from downloaded lwir and visible). Available options are {dataset_options_keys}. Usage: -c item1 item2, -c item3")
    parser.add_argument('-df', '--dataset-format', dest='dformat', type=str, default=dataset_keys[1],
                        help=f"Format of the dataset to be generated. One of the following: {dataset_keys}")
    
    opts = parser.parse_args()

    option_list_default = list(opts.olist)
    dataset_format = opts.dformat
    
    if dataset_format not in dataset_keys:
        raise KeyError(f"Dataset format provided ({dataset_format}) is not part of the ones avalable: {dataset_keys}.")

    log(f"Update datasets with options: {opts}")
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("Do you want to continue [Y/N]? ").lower()
    
    if answer == "y":
        checkDataset(option_list_default, dataset_format) # 'rgbt', 'hsvt' ... see rgb_thermal_mix.py for more info


