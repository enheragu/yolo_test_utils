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

from datetime import datetime


from yolo.utils import DEFAULT_CFG
import yolo.v8.detect as yolo_detc
from yolo.cfg import get_cfg
from ultralytics import YOLO

from config_utils import yolo_architecture_path, log, parseYaml, generateCFGFiles, clearCFGFIles, handleArguments

train_iteration = 0

if __name__ == '__main__':

    condition_list, option_list, model_list = handleArguments()

    start_time = datetime.now()

    dataset_config_list = generateCFGFiles(condition_list, option_list)
    for yolo_model in model_list: 
        for dataset in dataset_config_list:
            train_iteration += 1

            log("--------------------------------------------------------------------------")
            log(f"[{yolo_model}][test {train_iteration}] - Train based on {dataset} dataset:")
            data = parseYaml(dataset)
            log(f"\t · Train datasets: {data['train']} \n\t · Validation datasets: {data['val']} \n\t · Test datasets: {data['test']}")
            
            for model_option in ('train', 'test', 'val'):
                images = 0
                if data[model_option]:
                    for dataset_list in data[model_option]:
                        data_path = f"{data['path']}/{dataset_list}/images"
                        images += len(glob.glob1(data_path,"*.png"))
                    log(f"[{yolo_model}][test {train_iteration}] - {model_option.title()} with {images} images")
            
            dataset_start_time = datetime.now()

            args = {}
            args['mode'] = 'train'
            args['name'] = f'train_based_{yolo_model}/' + dataset.split("/")[-1].replace(".yaml","").replace("dataset_","")
            # args['pretrained']
            args['data'] = dataset
            args['pretrained'] = True
            args['model'] = yolo_model 
            # args['imgsz'] = 32
            args['epochs'] = 300
            args['batch'] = 16
            # args['save'] = False
            args['cache'] = 'ram'
            args['save_txt'] = True
            args['verbose'] = True
            args['save_conf'] = True
            args['device'] = 'cpu' #'0'
            
            trainer = yolo_detc.DetectionTrainer(overrides=args)
            try:
                trainer.train()
            except Exception as e:
                log(f'Exception caught: {e}')
                raise e
            
            log(f"Training and validation of model for {dataset} took {datetime.now() - dataset_start_time} (h/min/s)")
    
    clearCFGFIles(dataset_config_list)
    log("")
    log(f"Training process took {datetime.now() - start_time} (h/min/s)")
