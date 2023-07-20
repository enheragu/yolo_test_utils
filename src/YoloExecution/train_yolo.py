#!/usr/bin/env python3
# encoding: utf-8

"""
    Encapsulates YOLO training funtionality with custom dataset and logging handling
"""

from pathlib import Path
import glob
from datetime import datetime

from yolo.utils import DEFAULT_CFG
import yolo.v8.detect as yolo_detc
from yolo.cfg import get_cfg
from ultralytics import YOLO

from config_utils import dataset_config_path, log, parseYaml, generateCFGFiles, clearCFGFIles, handleArguments


def TestTrainYolo(condition_list, option_list, model_list, device, cache, pretrained):
    train_iteration = 0

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
                images_png = 0
                images_npy = 0
                if data[model_option]:
                    for dataset_list in data[model_option]:
                        data_path = f"{data['path']}/{dataset_list}/images"
                        images_png += len(glob.glob1(data_path,"*.png"))
                        images_npy += len(glob.glob1(data_path,"*.npy"))
                    if images_png:
                        log(f"[{yolo_model}][test {train_iteration}] - {model_option.title()} with {images_png} png images")
                    if images_npy:
                        log(f"[{yolo_model}][test {train_iteration}] - {model_option.title()} with {images_npy} npy images")
            
            dataset_start_time = datetime.now()

            args = {}
            args['mode'] = 'train'
            args['name'] = f'train_based_{yolo_model}/' + dataset.split("/")[-1].replace(".yaml","").replace("dataset_","")
            args['data'] = dataset
            args['model'] = yolo_model if ".yaml" not in yolo_model else f"{dataset_config_path}/{yolo_model}"  # If its a yaml check in configuration path
            # args['imgsz'] = 32
            args['epochs'] = 300
            args['batch'] = 16
            # args['save'] = False
            args['save_txt'] = True
            args['verbose'] = True
            args['save_conf'] = True

            args['device'] = device
            args['cache'] = cache
            args['pretrained'] = pretrained
            
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


if __name__ == '__main__':
    condition_list, option_list, model_list, device, cache, pretrained, _ = handleArguments()
    TestTrainYolo(condition_list, option_list, model_list, device, cache, pretrained)