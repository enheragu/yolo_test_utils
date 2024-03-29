#!/usr/bin/env python3
# encoding: utf-8



import os
from pathlib import Path
import sys
import logging
import argparse
import glob

import yaml
from yaml.loader import SafeLoader
from datetime import datetime

import ultralitics_yolov5.val as yolo_val

# python3 ultralitics-yolov5/val.py --weights yolov5s.pt --dataset_config_data dataset_config/yolo_obj_classes.yaml --img 640 --save-txt --verbose --save-conf --save-hybrid
MODEL_TO_RUN = ('yolov5s.pt')#, 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt')
dataset_config_yaml = 'dataset_config/yolo_obj_classes.yaml'
dataset_config_yaml_tmp = 'tmp_dataset_config.yaml'
MAX_FILE_PROCESS = 8000


DATASET_STATUS = {True:'processed', False:'ignored'}
already_run_yaml = "test_cache_run.yaml"
already_run = {} # Initialize data
for status in DATASET_STATUS.values():
    already_run[status] = set()

dataset_num = 0

def parseYaml(file_path):
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Battery validation test for YOLO with Kaist dataset')
    parser.add_argument('-cc', '--clear_cache', action='store_true', help="Clears cache dataset_config_data from previous test to perform all of them from the start")
    opt = vars(parser.parse_args())

    # If clear cache is active, remove file to perform all tests
    if opt['clear_cache']:
        if os.path.isfile(already_run_yaml):
            print("Cache file cleared to restart all tests")
            os.remove(already_run_yaml)

    # Parse dataset path to get dataset_config_data information
    dataset_config_data = parseYaml(dataset_config_yaml)
    dataset_path = dataset_config_data['path']
    
    # Remove option so that yolo arg parser does not fail
    if '--clear_cache' in sys.argv:
        sys.argv.remove('--clear_cache')

    start_time = datetime.now()

    # Iterate all models to make sure at the begining all are installed and downloaded
    # Make sure that is a list in case only one element is provided
    if type(MODEL_TO_RUN) is type(str()):
        MODEL_TO_RUN = [MODEL_TO_RUN]

    for yolo_model in MODEL_TO_RUN: 
        for dataset in os.listdir(dataset_path):
            for option in ("visible" , "lwir"):
                path_name = f"{yolo_model}/{dataset}_{option}/"
                
                if "train-night-20" not in path_name and "train-day-20" not in path_name:
                    print(f"[{yolo_model}][test {dataset_num}] {dataset}_{option} set to be ignored.")
                    status = DATASET_STATUS[False]
                    continue
                
                # Filter folders with too many images, based on computer resources
                # bigger datasets cannot be handled
                data_path = f"{dataset_path}/{dataset}/{option}/images"
                png_files = len(glob.glob1(data_path,"*.png"))
                if png_files > MAX_FILE_PROCESS:
                    print(f"[{yolo_model}][test {dataset_num}] {dataset}_{option} has {png_files} files, is to be ignored as is over {MAX_FILE_PROCESS}.")
                    status = DATASET_STATUS[False]
                    continue

                # # For now ignore big datasets
                # if "all" in path_name or "test-day-01" in path_name:
                #     print(f"[{yolo_model}][test {dataset_num}] {dataset}_{option} is configured to be ignored.")
                #     continue

                # If theres a cache file, check if this run was already performed and ignore it
                if os.path.isfile(already_run_yaml):
                    already_run = parseYaml(already_run_yaml)
                    for dataset_status_iter in DATASET_STATUS.values():
                        status = dataset_status_iter
                        already_run[status] = set(already_run[status])
                        if path_name in already_run[status]:
                            print(f"[{yolo_model}][test {dataset_num}] {dataset}_{option} is already in cache file marked as {status}.")
                            continue                
                
                # output_test_path = Path('./ultralitics_yolov5/runs/val/' + path_name)
                # output_test_path.mkdir(parents=True, exist_ok=True)
                # logging.basicConfig(filename = str(output_test_path) + "/std_out.log")

                dataset_num += 1
                dataset_start_time = datetime.now()
                print("-------------------------------------------------")
                print(f"[{yolo_model}][test {dataset_num}] Check {dataset}_{option} dataset with {png_files} images")
                
                # Tune manually the arguments for yolo run
                dataset_config_data["val"] = f"{dataset}/{option}/"
                dataset_config_data["train"] = f"{dataset}/{option}/"
                with open(dataset_config_yaml_tmp, "w+") as tmp_file:
                    yaml.dump(dataset_config_data, tmp_file)

                sys.argv.extend(['--weights', yolo_model])
                sys.argv.extend(['--data', dataset_config_yaml_tmp])
                sys.argv.extend(['--img', '640'])
                sys.argv.extend(['--save-txt'])
                sys.argv.extend(['--verbose'])
                sys.argv.extend(['--save-conf'])
                sys.argv.extend(['--save-json'])
                # sys.argv.extend(['--save-hybrid'])
                sys.argv.extend(['--name', path_name])

                opt = yolo_val.parse_opt()    
                yolo_val.main(opt)

                print(f"[{yolo_model}][test {dataset_num}]Dataset processing took {datetime.now() - dataset_start_time} (h/min/s)")
                print("-------------------------------------------------")
                status = DATASET_STATUS[True]

            # Update the already run instances
            already_run[status].add(path_name)
            with open(already_run_yaml, "w+") as file:
                yaml.dump(already_run, file) # cast to set to remove duplicates

            # Check that tmp file exist and clears it
            if os.path.isfile(dataset_config_yaml_tmp):
                os.remove(dataset_config_yaml_tmp)
            # print(f"[{yolo_model}][test {dataset_num}]Processed {dataset_num} datasets and took {datetime.now() - start_time} (h/min/s)")

    print()
    print(f"Processed {dataset_num} datasets and took {datetime.now() - start_time} (h/min/s)")