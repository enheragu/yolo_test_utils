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

from config_utils import dataset_config_path, parseYaml, generateCFGFiles, clearCFGFIles, handleArguments, yolo_output_path, dataset_tags_default
from log_utils import log, bcolors

from compress_label_folder import compress_output_labels


def TestTrainYolo(condition_list, option_list, model_list, device, cache, pretrained, path_name_in = None, dataset_tag = dataset_tags_default[0], batch = 16, deterministic = True):
    train_iteration = 0

    start_time = datetime.now()

    dataset_config_list = generateCFGFiles(condition_list, option_list, dataset_tag = dataset_tag)
    for yolo_model in model_list: 
        for dataset in dataset_config_list:
            train_iteration += 1

            log("--------------------------------------------------------------------------")
            log(f"[{yolo_model}][test {train_iteration}] - Train based on {dataset} dataset:")
            data = parseYaml(dataset)
            log(f"\t · Train datasets: {data['train']} \n\t · Validation datasets: {data['val']} \n\t · Test datasets: {data['test']}")
            
            yaml_data = {'n_images': {}}
            for model_option in ('train', 'test', 'val'):
                images_png = 0
                images_npy = 0
                if data[model_option]:
                    for dataset_list in data[model_option]:
                        data_path = f"{data['path']}/{dataset_list}/images"
                        images_png += len(glob.glob1(data_path,"*.png"))
                        images_npy += len(glob.glob1(data_path,"*.npy"))
                        images_npy += len(glob.glob1(data_path,"*.npz"))
                    if images_png:
                        log(f"[{yolo_model}][test {train_iteration}] - {model_option.title()} with {images_png} png images")
                    if images_npy:
                        log(f"[{yolo_model}][test {train_iteration}] - {model_option.title()} with {images_npy} npy images")
                    
                    yaml_data['n_images'][model_option] = images_png + images_npy

            dataset_start_time = datetime.now()
            
            if path_name_in is None:
                path_name = f'train_based_{yolo_model}/' + dataset.split("/")[-1].replace(".yaml","").replace("dataset_","") 
            else:
                path_name = path_name_in + "/" + dataset.split("/")[-1].replace(".yaml","").replace("dataset_","") 
            
            args = {}
            args['mode'] = 'train'
            args['name'] = path_name
            args['data'] = dataset
            args['model'] = yolo_model if ".yaml" not in yolo_model else f"{dataset_config_path}/{yolo_model}"  # If its a yaml check in configuration path
            # args['imgsz'] = 32
            args['epochs'] = 500
            args['batch'] = batch
            # args['save'] = False
            args['save_txt'] = True
            args['verbose'] = True
            args['save_conf'] = True
            args['patience'] = 10
            args['deterministic'] = deterministic

            args['device'] = device
            args['cache'] = cache
            args['pretrained'] = pretrained
                        
            yaml_data['pretrained'] = pretrained
            yaml_data['dataset_tag'] = dataset_tag
            
            trainer = yolo_detc.DetectionTrainer(overrides=args)
            try:
                trainer.train()
                with open(Path(trainer.save_dir) / f'results.yaml', 'a') as file:
                    import yaml
                    yaml.dump(yaml_data, file)

            except Exception as e:
                log(f'Exception caught: {e}')
                raise e
            
            
            log(f"Training and validation of model for {dataset} took {datetime.now() - dataset_start_time} (h/min/s)")
            compress_output_labels(trainer.save_dir)
    
    clearCFGFIles(dataset_config_list)
    log("")
    log(f"Complete training process took {datetime.now() - start_time} (h/min/s)")
    


if __name__ == '__main__':
    condition_list, option_list, model_list, device, cache, pretrained, _ = handleArguments()
    TestTrainYolo(condition_list, option_list, model_list, device, cache, pretrained)