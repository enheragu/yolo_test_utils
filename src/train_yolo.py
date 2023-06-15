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

from config import dataset_config_yaml, yolo_architecture_path


def parseYaml(file_path):
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)
    
if __name__ == '__main__':

    start_time = datetime.now()

    args = {}
    args['name'] = 'yolo-kaist-night'
    # args['pretrained']
    args['data'] = dataset_config_yaml
    args['pretrained'] = True
    args['model'] = 'yolov8x.pt' # pt file to transfer learn
    # args['model'] = yolo_architecture_path # YAML file to train
    # args['imgsz'] = 32
    args['epochs'] = 1
    args['batch'] = 16
    # args['save'] = False
    args['device'] = '0'
    
    trainer = yolo_detc.DetectionTrainer(overrides=args)
    try:
        trainer.train()
    except Exception as e:
        print(f'Expected exception caught: {e}')
    
    print()
    print(f"Training process took {datetime.now() - start_time} (h/min/s)")