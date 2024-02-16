#!/usr/bin/env python3
# encoding: utf-8

"""
    This script implements a fast test (CPU) to be able to test train / validation SW stack to test new implementations without
    all the time consuming execution. Runs in CPU to leave GPU free for a real test to be run/running in paralel.
    Tests YOLO SW!
"""
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


from config_utils import log, dataset_config_path
from YoloExecution.train_yolo import TestTrainYolo
from YoloExecution.validation_yolo import TestValidateYolo

from yolo.utils import DEFAULT_CFG
import yolo.v8.detect as yolo_detc
from yolo.cfg import get_cfg

yolo_model = 'yolov8x.pt'
dataset = f"{dataset_config_path}/dataset_fasttest.yaml"
cache = False
device = 'CPU'

if __name__ == '__main__':
    log(f"\n\n\n{'#'*30}\n[START FAST-TEST YOLO EXECUTION]")

    ## VALIDATION
    args = {} 
    args['mode'] = 'val'
    args['name'] = "fast_test/validate/"
    args['model'] = yolo_model
    args['data'] = dataset
    args['save_txt'] = True
    args['verbose'] = True
    args['save_conf'] = True
    args['save_json'] = True
    args['device'] = device

    args = get_cfg(cfg=DEFAULT_CFG, overrides=args)
    validator = yolo_detc.DetectionValidator(args=args)
    validator(model=args.model)


    ## TRAINING
    args = {}
    args['mode'] = 'train'
    args['name'] = "fast_test/train/"
    args['data'] = dataset
    args['model'] = yolo_model
    args['epochs'] = 1
    args['batch'] = 1
    args['save_txt'] = True
    args['verbose'] = True
    args['save_conf'] = True

    args['device'] = device
    args['cache'] = cache
    args['pretrained'] = True
    
    trainer = yolo_detc.DetectionTrainer(overrides=args)
    try:
        trainer.train()
    except Exception as e:
        log(f'Exception caught: {e}')
        raise e
    
    log(f"\n\n\n{'#'*30}\n[CLEAN FINISH TEST EXECUTION]")