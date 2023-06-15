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

from config import yolo_architecture_path, dataset_config_path, dataset_config_list, yolo_dataset_path, log

yolo_model = 'yolov8x.pt' # pt file to transfer learn
# yolo_model = yolo_architecture_path # YAML file to train

train_iteration = 0

def parseYaml(file_path):
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)
    
if __name__ == '__main__':

    start_time = datetime.now()

    for dataset in dataset_config_list:
        train_iteration += 1

        log("--------------------------------------------------------------------------")
        log(f"[{yolo_model}][test {train_iteration}] - Train based on {dataset} dataset:")
        data = parseYaml(dataset_config_path + dataset)
        log(f"\t · Train datasets: {data['train']} \n\t · Validation datasets: {data['val']} \n\t · Test datasets: {data['test']}")
        
        images = 0
        for train_dataset in data['train']:
            data_path = f"{yolo_dataset_path}/{train_dataset}/images"
            images += len(glob.glob1(data_path,"*.png"))
        log(f"[{yolo_model}][test {train_iteration}] - Train with {images} images")
        
        images = 0
        for val_dataset in data['val']:
            data_path = f"{yolo_dataset_path}/{val_dataset}/images"
            images += len(glob.glob1(data_path,"*.png"))
        log(f"[{yolo_model}][test {train_iteration}] - Validate with {images} images")


        dataset_start_time = datetime.now()

        args = {}
        args['mode'] = 'train'
        args['name'] = 'train_based_yolo8x.pt/' + dataset.replace(".yaml","").replace("dataset_","")
        # args['pretrained']
        args['data'] = dataset_config_path + dataset
        args['pretrained'] = True
        args['model'] = yolo_model 
        # args['imgsz'] = 32
        args['epochs'] = 300
        args['batch'] = 16
        # args['save'] = False
        args['device'] = '0'
        args['cache'] = True
        args['save_txt'] = True
        args['verbose'] = True
        args['save_conf'] = True
        
        trainer = yolo_detc.DetectionTrainer(overrides=args)
        try:
            trainer.train()
        except Exception as e:
            log(f'Exception caught: {e}')
            raise e
        
        log(f"Training and validation of model for {dataset} took {datetime.now() - dataset_start_time} (h/min/s)")
    
    log()
    log(f"Training process took {datetime.now() - start_time} (h/min/s)")