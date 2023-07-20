#!/usr/bin/env python3
# encoding: utf-8

from pathlib import Path
import glob
from datetime import datetime

from yolo.utils import DEFAULT_CFG
import yolo.v8.detect as yolo_detc
from yolo.cfg import get_cfg
from ultralytics import YOLO


from config_utils import yolo_output_path, log, parseYaml, generateCFGFiles, clearCFGFIles, handleArguments

    
def TestValidateYolo(condition_list, option_list, model_list, device, cache, pretrained):
    validation_iteration = 0
    start_time = datetime.now()

    dataset_config_list = generateCFGFiles(condition_list, option_list)
    for yolo_model in model_list: 
        for dataset in dataset_config_list:
            validation_iteration += 1

            path_name = "validate_" + yolo_model + "/" + dataset.split("/")[-1].replace(".yaml","").replace("dataset_","") + "/"           
            
            dataset_start_time = datetime.now()
            log("--------------------------------------------------------------------------")
            log(f"[{yolo_model}][test {validation_iteration}] - Check {dataset} dataset")
            data = parseYaml(dataset)
            log(f"[{yolo_model}][test {validation_iteration}] - Validation datasets: {data['val']}")      
            
            images_png = 0
            images_npy = 0
            for val_dataset in data['val']:
                data_path = f"{data['path']}/{val_dataset}/images"
                images_png += len(glob.glob1(data_path,"*.png"))
                images_npy += len(glob.glob1(data_path,"*.npy"))
            if images_png:
                log(f"[{yolo_model}][test {validation_iteration}] - Validation with {images_png} png images")
            if images_npy:
                log(f"[{yolo_model}][test {validation_iteration}] - Validation with {images_npy} npy images")
            


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
            args['device'] = device
            # args['save_hybrid'] = True -> PROBLEMS WITH TENSOR SIZE

            args = get_cfg(cfg=DEFAULT_CFG, overrides=args)
            # opt = yolo_detc.parse_opt()    
            # log(DEFAULT_CFG)
            # yolo_detc.val()
            validator = yolo_detc.DetectionValidator(args=args)
            validator(model=args.model)


            log(f"[{yolo_model}][test {validation_iteration}] - Dataset processing took {datetime.now() - dataset_start_time} (h/min/s)")
            log("-------------------------------------------------")
           
    clearCFGFIles(dataset_config_list)
    log()
    log(f"Processed {validation_iteration} datasets and took {datetime.now() - start_time} (h/min/s)")


if __name__ == '__main__':
    condition_list, option_list, model_list, device, cache, pretrained, _ = handleArguments()
    TestValidateYolo(condition_list, option_list, model_list, device, cache, pretrained)