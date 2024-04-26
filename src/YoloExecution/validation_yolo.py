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


from utils import parseYaml, dumpYaml, log, bcolors, getGPUTestID
from Dataset import generateCFGFiles, clearCFGFIles, dataset_tags_default
from argument_parser import handleArguments, yolo_outpu_log_path

def TestValidateYolo(condition_list, option_list, model_list, device, cache, pretrained, path_name_in = None, dataset_tag = dataset_tags_default[0]):
    validation_iteration = 0
    start_time = datetime.now()

    dataset_config_list = generateCFGFiles(condition_list, option_list, dataset_tag = dataset_tag)
    for yolo_model in model_list: 
        for dataset in dataset_config_list:
            validation_iteration += 1
            

            id = getGPUTestID()

            if path_name_in is None:
                path_name = "validate_" + yolo_model + "/" + dataset.split("/")[-1].replace(".yaml","").replace("dataset_","") + id
            else:
                path_name = path_name_in + "/"  + dataset.split("/")[-1].replace(".yaml","").replace("dataset_","") + id

            dataset_start_time = datetime.now()
            log("-------------------------------------")
            log(f"[{yolo_model}][test {validation_iteration}] - Check {dataset} dataset")
            data = parseYaml(dataset)
            log(f"[{yolo_model}][test {validation_iteration}] - Validation datasets: {data['val']}")      
            
            images_png = 0
            images_npy = 0
            yaml_data = {'n_images': {}}
            for val_dataset in data['val']:
                data_path = f"{data['path']}/{val_dataset}/images"
                images_png += len(glob.glob1(data_path,"*.png"))
                images_npy += len(glob.glob1(data_path,"*.npy"))
                images_npy += len(glob.glob1(data_path,"*.npz"))
            if images_png:
                log(f"[{yolo_model}][test {validation_iteration}] - Validation with {images_png} png images")
            if images_npy:
                log(f"[{yolo_model}][test {validation_iteration}] - Validation with {images_npy} npy images")            

            yaml_data['n_images']['val'] = images_png + images_npy
            yaml_data['dataset_tag'] = dataset_tag

            args = {} 
            # args['project'] = 'detection'
            args['mode'] = 'val'
            args['name'] = path_name + "_" + datetime.now().strftime("%Y%m%d")
            args['model'] = yolo_model
            args['data'] = dataset
            # args['imgsz'] = 640
            args['save_txt'] = True
            args['verbose'] = True
            args['save_conf'] = True
            args['save_json'] = True
            args['device'] = device
            # args['save_hybrid'] = True -> PROBLEMS WITH TENSOR SIZE

            args = get_cfg(cfg=DEFAULT_CFG, overrides=args)
            # opt = yolo_detc.parse_opt()    
            # log(DEFAULT_CFG)
            # yolo_detc.val()
            validator = yolo_detc.DetectionValidator(args=args)
            validator(model=args.model)

            dumpYaml(Path(validator.save_dir) / f'results.yaml', yaml_data)

            log(f"[{yolo_model}][test {validation_iteration}] - Dataset processing took {datetime.now() - dataset_start_time} (h/min/s)")
            log("-------------------------------------")
           
    clearCFGFIles(dataset_config_list)
    # log()
    log(f"Processed {validation_iteration} datasets and took {datetime.now() - start_time} (h/min/s)")


if __name__ == '__main__':
    condition_list, option_list, model_list, device, cache, pretrained, _ = handleArguments()
    TestValidateYolo(condition_list, option_list, model_list, device, cache, pretrained)