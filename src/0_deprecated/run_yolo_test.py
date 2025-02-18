#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import traceback

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


from argument_parser import handleArguments
from utils import log, bcolors
from YoloExecution.train_yolo import TestTrainYolo
from YoloExecution.validation_yolo import TestValidateYolo
from update_datset import checkDataset


if __name__ == '__main__':
    log(f"\n\n\n{'#'*30}\n[START TEST EXECUTION]")
    finish_ok = True
    try:
        log(f"Arguments from system: {sys.argv = }")
        condition_list, option_list, model_list, opts = handleArguments()
        checkDataset(option_list, opts.dformat)

        for mode in opts.run_mode:
            if mode == 'val':
                TestValidateYolo(condition_list, option_list, model_list, opts.device, opts.cache, opts.pretrained, opts.path_name, opts.dformat)
            elif mode == 'train':
                TestTrainYolo(condition_list, option_list, model_list, opts.device, opts.cache, opts.pretrained, opts.path_name, opts.dformat, opts.batch)

    except KeyError as e:
        log(f"Options failed were:\n\t· {condition_list = }\n\t· {option_list = }\n\t· {model_list = };\n\t· run mode: {opts.run_mode}", bcolors.ERROR)
        log(f"Catched exception: {e}", bcolors.ERROR)
        log(traceback.format_exc(), bcolors.ERROR)

        if 'NTFY_TOPIC' in os.environ:
            finish_ok = False
            import requests
            topic = os.getenv('NTFY_TOPIC')

            raw_msg = f"Options executed were: {condition_list = }; {option_list = }; {model_list = }; run mode = {opts.run_mode}\n"
            raw_msg += f"Catched exception: {e}"
            requests.post(f"https://ntfy.sh/{topic}", data =str(raw_msg).encode(encoding='utf-8'),
                        headers={"Title": "Training execution failed", "Priority": "high", "Tags": "-1,man_facepalming"
                        })

    if finish_ok:
        log(f"Options executed were:\n\t· {condition_list = }\n\t· {option_list = }\n\t· {model_list = };\n\t· run mode: {opts.run_mode}")
        log(f"\n\n\n{'#'*30}\n[CLEAN FINISH TEST EXECUTION]")


        if 'NTFY_TOPIC' in os.environ:
            import requests
            topic = os.getenv('NTFY_TOPIC')

            raw_msg = f"Options executed were: {condition_list = }; {option_list = }; {model_list = }; run mode = {opts.run_mode}"
            requests.post(f"https://ntfy.sh/{topic}", data =str(raw_msg).encode(encoding='utf-8'),
                        headers={"Title": "Training execution finished", "Priority": "default", "Tags": "+1,partying_face"
                        })