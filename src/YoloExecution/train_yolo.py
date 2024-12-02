#!/usr/bin/env python3
# encoding: utf-8

"""
    Encapsulates YOLO training funtionality with custom dataset and logging handling
"""
import os

from pathlib import Path
import glob
from datetime import datetime

from yolo.utils import DEFAULT_CFG
import yolo.v8.detect as yolo_detc
from yolo.cfg import get_cfg
from ultralytics import YOLO

from utils import parseYaml, dumpYaml, log, bcolors
from Dataset import dataset_config_path

from compress_label_folder import compress_output_labels


def TestTrainYolo(dataset, yolo_model, path_name, opts):
    start_time = datetime.now()


    log("-------------------------------------")
    log(f"[{yolo_model}] - Train based on {dataset} dataset:")
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
                log(f"[{yolo_model}] - {model_option.title()} with {images_png} png images")
            if images_npy:
                log(f"[{yolo_model}] - {model_option.title()} with {images_npy} npy images")
            
            yaml_data['n_images'][model_option] = images_png + images_npy
    

    args = {}
    args['mode'] = 'train'
    args['name'] = path_name
    args['data'] = dataset
    args['model'] = yolo_model if ".yaml" not in yolo_model else f"{dataset_config_path}/{yolo_model}"  # If its a yaml check in configuration path
    # args['imgsz'] = 32
    args['epochs'] = 500
    args['batch'] = opts.batch
    args['save'] = True
    # args['save_period'] = 2
    args['save_json'] = True
    # args['save_hybrid'] = True

    args['save_txt'] = True
    args['verbose'] = True
    args['save_conf'] = True
    args['iou'] = 0.5
    args['patience'] = 10
    args['deterministic'] = opts.deterministic

    args['device'] = opts.device
    args['cache'] = opts.cache
    args['pretrained'] = opts.pretrained
    args['seed'] = 1
    args['resume'] = opts.resume
    yaml_data['seed'] = 1
                
    yaml_data['pretrained'] = opts.pretrained
    yaml_data['dataset_tag'] = opts.dformat
    yaml_data['thermal_equalization'] = opts.thermal_eq
    yaml_data['rgb_equalization'] = opts.rgb_eq
    yaml_data['resume'] = opts.resume
    
    trainer = yolo_detc.DetectionTrainer(overrides=args)
    try:
        trainer.train()
        dumpYaml(Path(trainer.save_dir) / f'results.yaml', yaml_data, 'a')

    except Exception as e:
        log(f'Exception caught: {e}')
        raise e
    
    
    log(f"Training and validation of model for {dataset} took {datetime.now() - start_time} (h/min/s)")
    compress_output_labels(trainer.save_dir)
    log("-------------------------------------")



if __name__ == '__main__':
    pass
    # condition_list, option_list, model_list, device, cache, pretrained, _ = handleArguments()
    # TestTrainYolo(condition_list, option_list, model_list, device, cache, pretrained)