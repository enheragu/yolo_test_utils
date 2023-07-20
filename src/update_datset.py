#!/usr/bin/env python3
# encoding: utf-8

"""
    Checks that dataset exists with all options needed. If not downloads Kaist, prepares YOLO annotated version 
    (see kaist_to_yolo_annotations.py) and extras (see rgb_thermal_mix.py)
"""

import os

from pathlib import Path
import requests
import xtarfile as tarfile
from clint.textui import progress
from tqdm import tqdm

from argparse import ArgumentParser

from Dataset.kaist_to_yolo_annotations import kaistToYolo
from Dataset.rgb_thermal_mix import dataset_options, make_dataset

from config_utils import kaist_path, yolo_dataset_path, log


def getKaistData():
    filename="kaist-cvpr15.tar.gz"
    url="https://onedrive.live.com/download?cid=1570430EADF56512&resid=1570430EADF56512%21109419&authkey=AJcMP-7Yp86PWoE"
    download_path = f"{kaist_path}/../{filename}"

    # make sure that kaist path exists
    Path(kaist_path).mkdir(parents=True, exist_ok=True)

    log(f"Getting Kaist dataset from source. Downloading it to {download_path}. (Note that it can take quite some time).")
    r = requests.get(url, stream=True)
    with open(download_path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
            if chunk:
                f.write(chunk)
                f.flush()


    log(f"Extracting data from {download_path}. (Note that it can take up to 10-20 minutes).")
    # with tarfile.open(download_path, 'r') as archive:
    #     archive.extractall()

    with tarfile.open(download_path, 'r') as tar:
        # Go over each member
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            # Extract member
            tar.extract(member=member)

def checkKaistDataset(options = []):
    # Ensure input is a list
    if type(options) is not type(list()):
        options = [options]

    # Check if kaist dataset is already in the system
    if not os.path.exists(f"{kaist_path}/images"):
        log(f"Kaist dataset could not be found in {kaist_path}. Downloading it from scratch.")
        getKaistData()
    else:
        log(f"Kaist dataset found in {kaist_path}, no need to re-download.")
    
    # Check that YOLO version exists or create it
    setfolders = [ f.path for f in os.scandir(yolo_dataset_path) if f.is_dir() ]
    options_found = [ f.name for f in os.scandir(setfolders[0]) if f.is_dir() ]
    if 'lwir' not in options_found and 'visible' not in options_found:
        log(f"Kaist-YOLO dataset could not be found in {yolo_dataset_path}. Generating new labeling for both lwir and visible sets.")
        kaistToYolo()
    else:
        log(f"Kaist-YOLO dataset found in {yolo_dataset_path}, no need to re-label it.")

    # Check that needed versions exist or create them
    for option in options:
        if option not in options_found:
            log(f"Custom dataset for option {option} requested but not found in dataset folders. Generating it.")
            make_dataset(option)
        else:
            log(f"Custom dataset for option {option} requested is already in dataset folder.")
            
if __name__ == '__main__':
    parser = ArgumentParser(description="Checks if kaist dataset exists in expected location and generates it if not (download, extract, reformat or regenerate).")
    parser.add_argument('-o', '--option', action='store', dest='olist',
                        type=str, nargs='*', default=dataset_options.keys(),
                        help=f"Extra options of the datasets to be included (apart from downloaded lwir and visible). Available options are {dataset_options.keys()}. Usage: -c item1 item2, -c item3")
    opts = parser.parse_args()

    option_list_default = list(opts.olist)
    checkKaistDataset(option_list_default) # 'rgbt', 'hsvt' ... see rgb_thermal_mix.py for more info