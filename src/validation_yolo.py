#!/usr/bin/env python3
# encoding: utf-8

import os
import sys

ultralitics_rel_path = (os.path.dirname(os.path.abspath(__file__)) + '/../ultralitics_yolov8/ultralytics/',
                        os.path.dirname(os.path.abspath(__file__)) + '/../ultralitics_yolov8/',
                        os.path.dirname(os.path.abspath(__file__)) + '/../'
                        )

for add_path in ultralitics_rel_path:
    try:
        sys.path.index(add_path) # Or os.getcwd() for this directory
    except ValueError:
        sys.path.append(add_path) # Or os.getcwd() for this directory


from pathlib import Path
import logging
import argparse
import glob

import yaml
from yaml.loader import SafeLoader
from datetime import datetime


from yolo.utils import DEFAULT_CFG
import yolo.v8.detect as yolo_detc
from yolo.cfg import get_cfg
from ultralytics import YOLO


from config_utils import yolo_dataset_path, generateCFGFiles, clearCFGFIles, yolo_output_path, log

# python3 ultralitics-yolov5/val.py --weights yolov5s.pt --dataset_config_data dataset_config/yolo_obj_classes.yaml --img 640 --save-txt --verbose --save-conf --save-hybrid
MODEL_TO_RUN = ('yolov8x.pt') #('yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')


DATASET_STATUS = {True:'processed', False:'ignored'}
already_run_yaml = f"{yolo_output_path}/test_cache_run.yaml"
already_run = {} # Initialize data
for status in DATASET_STATUS.values():
    already_run[status] = set()

validation_iteration = 0

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
            log("Cache file cleared to restart all tests")
            os.remove(already_run_yaml)
    
    # Remove option so that yolo arg parser does not fail
    if '--clear_cache' in sys.argv:
        sys.argv.remove('--clear_cache')


    # Iterate all models to make sure at the begining all are installed and downloaded
    # Make sure that is a list in case only one element is provided
    if type(MODEL_TO_RUN) is type(str()):
        MODEL_TO_RUN = [MODEL_TO_RUN]

    start_time = datetime.now()

    dataset_config_list = generateCFGFiles()
    for yolo_model in MODEL_TO_RUN: 
        for dataset in dataset_config_list:
            validation_iteration += 1

            path_name = "validate_" + yolo_model + "/" + dataset.split("/")[-1].replace(".yaml","").replace("dataset_","") + "/"
                    
            # If theres a cache file, check if this run was already performed and ignore it
            if os.path.isfile(already_run_yaml):
                already_run = parseYaml(already_run_yaml)
                for dataset_status_iter in DATASET_STATUS.values():
                    status = dataset_status_iter
                    already_run[status] = set(already_run[status])
                    if path_name in already_run[status]:
                        log(f"[{yolo_model}][test {validation_iteration}] {dataset} is already in cache file marked as {status}.")
                        continue                
            
            dataset_start_time = datetime.now()
            log("--------------------------------------------------------------------------")
            log(f"[{yolo_model}][test {validation_iteration}] - Check {dataset} dataset")
            data = parseYaml(dataset)
            log(f"[{yolo_model}][test {validation_iteration}] - Validation datasets: {data['val']}")
            
            images = 0
            for val_dataset in data['val']:
                data_path = f"{yolo_dataset_path}/{val_dataset}/images"
                images += len(glob.glob1(data_path,"*.png"))
            log(f"[{yolo_model}][test {validation_iteration}] - Validation with {images} images")
            


            args = {} 
            # args['project'] = 'detection'
            args['mode'] = 'val'
            args['name'] = path_name
            args['model'] = yolo_model
            args['data'] = dataset
            # args['imgsz'] = 640
            args['save_txt'] = True
            args['verbose'] = True
            args['save_conf'] = True
            args['save_json'] = True
            args['device'] = '0'
            # args['save_hybrid'] = True -> PROBLEMS WITH TENSOR SIZE

            args = get_cfg(cfg=DEFAULT_CFG, overrides=args)
            # opt = yolo_detc.parse_opt()    
            # log(DEFAULT_CFG)
            # yolo_detc.val()
            validator = yolo_detc.DetectionValidator(args=args)
            validator(model=args.model)


            log(f"[{yolo_model}][test {validation_iteration}] - Dataset processing took {datetime.now() - dataset_start_time} (h/min/s)")
            log("-------------------------------------------------")
            status = DATASET_STATUS[True]

            # Update the already run instances
            already_run[status].add(path_name)
            with open(already_run_yaml, "w+") as file:
                yaml.dump(already_run, file) # cast to set to remove duplicates
                # log(f"stored data in already_run_file ({already_run_yaml}); {already_run}")

            # log(f"[{yolo_model}][test {validation_iteration}]Processed {validation_iteration} datasets and took {datetime.now() - start_time} (h/min/s)")

    clearCFGFIles(dataset_config_list)
    log()
    log(f"Processed {validation_iteration} datasets and took {datetime.now() - start_time} (h/min/s)")