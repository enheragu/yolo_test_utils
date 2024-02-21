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

from log_utils import log, bcolors
from Dataset import kaistToYolo, dataset_options, make_dataset
from config_utils import kaist_path, kaist_yolo_dataset_path, dataset_tags_default


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

def checkKaistDataset(options = [], dataset_format = 'kaist_coco'):
    # Ensure input is a list
    if type(options) is not type(list()):
        options = [options]

    # Check if kaist dataset is already in the system
    if not os.path.exists(f"{kaist_path}/images"):
        log(f"[UpdateDataset::checkKaistDataset] Kaist dataset could not be found in {kaist_path}. Downloading it from scratch.")
        getKaistData()
    else:
        log(f"[UpdateDataset::checkKaistDataset] Kaist dataset found in {kaist_path}, no need to re-download.")
    
    # make sure that kaist-yolo path exists
    Path(kaist_yolo_dataset_path).mkdir(parents=True, exist_ok=True)

    # Check that YOLO version exists or create it
    setfolders = [ f.path for f in os.scandir(kaist_yolo_dataset_path) if f.is_dir() ]
    options_found = [ f.name for f in os.scandir(setfolders[0]) if f.is_dir() ] if setfolders else []
    
    # log(f"{kaist_yolo_dataset_path = }")
    # log(f"[UpdateDataset::checkKaistDataset] {setfolders = };\n{options_found =}\n")

    if 'lwir' not in options_found and 'visible' not in options_found:
        log(f"[UpdateDataset::checkKaistDataset] Kaist-YOLO dataset could not be found in {kaist_yolo_dataset_path}. Generating new labeling for both lwir and visible sets.")
        kaistToYolo(dataset_format)
        # Update with new options
        setfolders = [ f.path for f in os.scandir(kaist_yolo_dataset_path) if f.is_dir() ]
        options_found = [ f.name for f in os.scandir(setfolders[0]) if f.is_dir() ] if setfolders else []
    else:
        log(f"[UpdateDataset::checkKaistDataset] Kaist-YOLO dataset found in {kaist_yolo_dataset_path}, no need to re-label it.")
    
    # log(f"{kaist_yolo_dataset_path = }")
    # log(f"[UpdateDataset::checkKaistDataset] {setfolders = };\n{options_found =}\n")

    # Check that needed versions exist or create them
    for option in options:
        if option not in options_found:
            log(f"[UpdateDataset::checkKaistDataset] Custom dataset for option {option} requested but not found in dataset folders. Generating it.")
            make_dataset(option)
        else:
            log(f"[UpdateDataset::checkKaistDataset] Custom dataset for option {option} requested is already in dataset folder.")
            
if __name__ == '__main__':
    parser = ArgumentParser(description="Checks if kaist dataset exists in expected location and generates it if not (download, extract, reformat or regenerate).")
    parser.add_argument('-o', '--option', action='store', dest='olist',
                        type=str, nargs='*', default=dataset_options.keys(),
                        help=f"Extra options of the datasets to be included (apart from downloaded lwir and visible). Available options are {dataset_options.keys()}. Usage: -c item1 item2, -c item3")
    parser.add_argument('-df', '--dataset-format', dest='dformat', type=str, default=dataset_tags_default[0],
                        help=f"Format of the dataset to be generated. One of the following: {dataset_tags_default}")
    
    opts = parser.parse_args()

    option_list_default = list(opts.olist)
    dataset_format = opts.dformat
    
    if dataset_format not in dataset_tags_default:
        raise KeyError(f"Dataset format provided ({dataset_format}) is not part of the ones avalable: {dataset_tags_default}.")

    checkKaistDataset(option_list_default, dataset_format) # 'rgbt', 'hsvt' ... see rgb_thermal_mix.py for more info