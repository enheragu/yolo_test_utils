#!/usr/bin/env python3
# encoding: utf-8

import os 
from pathlib import Path
import glob
from datetime import datetime

from yolo.utils import DEFAULT_CFG
import yolo.v8.detect as yolo_detc
from yolo.cfg import get_cfg
from ultralytics import YOLO


from utils import parseYaml, dumpYaml, log, bcolors

def TestValidateYolo(dataset, yolo_model, path_name, opts, log_file_path):
    start_time = datetime.now()

    log("-------------------------------------")
    log(f"[{yolo_model}] - Check {dataset} dataset")
    data = parseYaml(dataset)
    log(f"[{yolo_model}] - Validation datasets: {data['val']}")      
    
    images_png = 0
    images_npy = 0
    yaml_data = {'n_images': {}}
    for val_dataset in data['val']:
        data_path = f"{data['path']}/{val_dataset}/images"
        images_png += len(glob.glob1(data_path,"*.png"))
        images_npy += len(glob.glob1(data_path,"*.npy"))
        images_npy += len(glob.glob1(data_path,"*.npz"))
    if images_png:
        log(f"[{yolo_model}] - Validation with {images_png} png images")
    if images_npy:
        log(f"[{yolo_model}] - Validation with {images_npy} npy images")            

    yaml_data['n_images']['val'] = images_png + images_npy
    yaml_data['dataset_tag'] = opts.dformat
    yaml_data['thermal_equalization'] = opts.thermal_eq
    yaml_data['rgb_equalization'] = opts.rgb_eq

    args = {} 
    # args['project'] = 'detection'
    args['mode'] = 'val'
    args['name'] = path_name
    args['model'] = yolo_model
    args['data'] = dataset
    # args['imgsz'] = 640
    args['save'] = True
    args['save_txt'] = True
    args['verbose'] = True
    args['save_conf'] = True
    args['save_json'] = True
    args['device'] = opts.device
    # args['save_hybrid'] = True -> PROBLEMS WITH TENSOR SIZE

    yaml_data['output_log_file'] = log_file_path

    args = get_cfg(cfg=DEFAULT_CFG, overrides=args)
    # opt = yolo_detc.parse_opt()    
    # log(DEFAULT_CFG)
    # yolo_detc.val()
    validator = yolo_detc.DetectionValidator(args=args)
    validator(model=args.model)

    dumpYaml(Path(validator.save_dir) / f'results.yaml', yaml_data, 'a')

    log(f"[{yolo_model}] - Dataset processing took {datetime.now() - start_time} (h/min/s)")
    log("-------------------------------------")
           

if __name__ == '__main__':
    pass
    # condition_list, option_list, model_list, device, cache, pretrained, _ = handleArguments()
    # TestValidateYolo(condition_list, option_list, model_list, device, cache, pretrained)