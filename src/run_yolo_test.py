#!/usr/bin/env python3
# encoding: utf-8

import os
import sys

ultralitics_rel_path = (os.path.dirname(os.path.abspath(__file__)) + '/../ultralitics_yolov8/ultralytics/',
                        os.path.dirname(os.path.abspath(__file__)) + '/../ultralitics_yolov8/',
                        os.path.dirname(os.path.abspath(__file__)) + '/../'
                        )
yolo_test_paths = (os.path.dirname(os.path.abspath(__file__)) + '/Dataset',
                   os.path.dirname(os.path.abspath(__file__)) + '/YoloExecution'
                  )

for add_path in ultralitics_rel_path + yolo_test_paths:
    try:
        sys.path.index(add_path) # Or os.getcwd() for this directory
    except ValueError:
        sys.path.append(add_path) # Or os.getcwd() for this directory


from config_utils import log, handleArguments
from YoloExecution.train_yolo import TestTrainYolo
from YoloExecution.validation_yolo import TestValidateYolo
from update_datset import checkKaistDataset


if __name__ == '__main__':
    log(f"\n\n\n{'#'*30}\n[START TEST EXECUTION]")
    condition_list, option_list, model_list, device, cache, pretrained, opts = handleArguments()

    checkKaistDataset(option_list)

    for mode in opts.run_mode:
        if mode == 'val':
            TestValidateYolo(condition_list, option_list, model_list, device, cache, pretrained)
        elif mode == 'train':
            TestTrainYolo(condition_list, option_list, model_list, device, cache, pretrained)
    
    log(f"\n\n\n{'#'*30}\n[CLEAN FINISH TEST EXECUTION]")