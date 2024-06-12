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

from utils import parseYaml, dumpYaml, log, bcolors, getGPUTestID
from Dataset import generateCFGFiles, clearCFGFIles, dataset_tags_default, dataset_config_path
from argument_parser import handleArguments, yolo_outpu_log_path

from compress_label_folder import compress_output_labels


def TestTrainYolo(dataset, model_list, opts):
    train_iteration = 0

    start_time = datetime.now()

    for yolo_model in model_list: 
    
        train_iteration += 1

        log("-------------------------------------")
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
        

        id = getGPUTestID()
        if opts.test_name is None:
            test_name = dataset.split("/")[-1].replace(".yaml","").replace("dataset_","") + id
        else:
            test_name = opts.test_name + id

        if opts.path_name is None:
            path_name = f'train_based_{yolo_model}/{test_name}'
        else:
            path_name = opts.path_name + "/" + test_name
        
        args = {}
        args['mode'] = 'train'
        args['name'] = path_name + "_" + datetime.now().strftime("%Y%m%d")
        args['data'] = dataset
        args['model'] = yolo_model if ".yaml" not in yolo_model else f"{dataset_config_path}/{yolo_model}"  # If its a yaml check in configuration path
        # args['imgsz'] = 32
        args['epochs'] = 500
        args['batch'] = opts.batch
        # args['save'] = False
        # args['save_json'] = True
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
                    
        yaml_data['pretrained'] = opts.pretrained
        yaml_data['dataset_tag'] = opts.dformat
        yaml_data['thermal_equalization'] = opts.thermal_eq
        
        trainer = yolo_detc.DetectionTrainer(overrides=args)
        try:
            trainer.train()
            dumpYaml(Path(trainer.save_dir) / f'results.yaml', yaml_data, 'a')

        except Exception as e:
            log(f'Exception caught: {e}')
            raise e
        
        
        log(f"Training and validation of model for {dataset} took {datetime.now() - dataset_start_time} (h/min/s)")
        compress_output_labels(trainer.save_dir)
        log("-------------------------------------")

    log("")
    log(f"Complete training process took {datetime.now() - start_time} (h/min/s)")
    


if __name__ == '__main__':
    pass
    # condition_list, option_list, model_list, device, cache, pretrained, _ = handleArguments()
    # TestTrainYolo(condition_list, option_list, model_list, device, cache, pretrained)