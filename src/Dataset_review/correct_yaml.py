#!/usr/bin/env python3
# encoding: utf-8

from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures

from GUI.dataset_manager import find_results_file, background_load_data
from utils import parseYaml
from utils import log, bcolors


###
# CUIDADO CON LANZAR ESTE SCRIPT
###

def wait_for_completion(futures):
    for future in as_completed(futures):
        yield future.result()
        
if __name__ == "__main__":
    datasets = find_results_file(ignored=False)
    parsed = {}

    # log(f"Found datasets to parse: {datasets.keys()}")
    with ProcessPoolExecutor() as executor:
        # futures = {key: executor.submit(background_load_data, key) for key in datasets}
        futures = {key: executor.submit(parseYaml, datasets[key]['path']) for key in datasets}   
    
        for future in wait_for_completion(futures.values()):
            pass

        for key in datasets:
            data = futures[key].result() 
            if not 'validation_0' in data:
                log(f"Did not find 'validation_0' tag in {key} and wont be processed.", bcolors.WARNING)
                continue
            elif not 'train_data' in data:
                log(f"Test still in progress. Did not find 'train_data' tag in {key} and wont be processed.", bcolors.WARNING)
                continue
            # key = data['validation_0']['name']
            parsed[key] = data

    log(f"Paralel parsing finished")

    for dataset in parsed:
        data = parsed[dataset]
        # log(f"Dataset {dataset} found in {datasets[dataset]['path']}")

        update = False
        # if not 'val_epoch' in data:
        #     log(f"'val_epoch' key not found in {dataset}", bcolors.WARNING)
        #     continue
        # val_tag = f"validation_{data['val_epoch']-1}"
        # if not val_tag in data:
        #     log(f"'{val_tag}' key not found in {dataset}", bcolors.WARNING)
        #     continue


        # if 'all' in dataset:
        #     # if 'n_images' in data:
        #     #     log(f"{dataset} dataset has {data['n_images']}", bcolors.WARNING)
        #     if 'n_images' not in data:
        #         data['n_images'] = {'train': 27586, 'val': 45140}
        #         log(f"Update n_images of {dataset}", bcolors.WARNING)
        #         update = True
        
        # if 'variance_day_visible_trained/day_visible' in dataset:
            # if 'n_images' in data:
            #     log(f"{dataset} dataset has {data['n_images']}", bcolors.WARNING)
            # if 'n_images' not in data:
            #     data['n_images'] = {'train': 10004, 'val': 30633}
            #     log(f"Update n_images of {dataset}", bcolors.WARNING)
            #     update = True

        # if 'trained' in dataset or 'train_based_yoloCh2x.yaml' in dataset:
        #     if 'pretrained' not in data:
        #         data['pretrained'] = False
        #         log(f"Update pretrained of {dataset}", bcolors.WARNING)
        #         update = True

        # if 'pretrained' in dataset or 'train_based_yolov8x.pt' in dataset:
        #     if 'pretrained' not in data:
        #         data['pretrained'] = True
        #         log(f"Update pretrained of {dataset}", bcolors.WARNING)
        #         update = True

        data['system_data'] = {
            'YOLO_v': '8.0.112',
            'device_type': '(NVIDIA GeForce RTX 4090, 24217MiB)',
            'python_v': '"3.8.10 (default, Nov 22 2023, 10:22:35) \n[GCC 9.4.0]"',
            'torch_v': '2.0.1+cu117'}
        update = True

        log(f"Update system_data of {dataset}", bcolors.WARNING)


        if 'yoloCh3x.yaml' in dataset:
            log(f"'yoloCh3x.yaml' update pretrained to False", bcolors.WARNING)
            update = True
            data['pretrained'] = False

        # if 'n_images' in data:
        #     train_images_total = data['n_images']['train']
        #     val_images_total = data['n_images']['val']

        #     val_images = data[val_tag]['data']['all']['Images']

        #     if val_images != val_images_total:
        #         log(f"Difference in images in {dataset}: \t\t{val_images = } vs {val_images_total = }; {train_images_total = }", bcolors.WARNING)

        #         data['n_images']['train'] = train_images_total/2
        #         data['n_images']['val'] = val_images_total/2
        #         update = True
        # else:
        #     log(f"'n_images' key not found in {dataset}", bcolors.ERROR)
        #     update = True
        #     data['n_images'] = {}
        #     if "night" in dataset:
        #         data['n_images']['train'] = 25176/2
        #         data['n_images']['val'] = 16759
        #     else:
        #         data['n_images']['train'] = 20008/2
        #         data['n_images']['val'] = 30633
                

        # if not 'pretrained' in data:
        #     log(f"'pretrained' key not found in {dataset}", bcolors.WARNING)
        #     update = True
        #     if 'yolov8x.pt' in dataset:
        #         data['pretrained'] = True
        #     else:
        #         data['pretrained'] = False

        # if 'yoloCh3x.yaml' in dataset:
        #     log(f"'yoloCh3x.yaml' update pretrained to False", bcolors.WARNING)
        #     update = True
        #     data['pretrained'] = False
            
        if update:
            with open(datasets[dataset]['path'], 'w') as file:
                import yaml
                yaml.dump(data, file)
            log(f"Updated {dataset}", bcolors.OKGREEN)
        else:
            log(f"No need to update {dataset}")