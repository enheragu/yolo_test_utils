
#!/usr/bin/env python3
# encoding: utf-8
"""
    Handles the loading of dataset data to be used later by the GUY
"""


import os
import sys
import time
import shutil

from datetime import datetime

import threading
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import csv

from config_utils import yolo_output_path as test_path
from config_utils import log, bcolors, parseYaml, dumpYaml

data_file_name = "results.yaml"
cache_path = f"{os.getenv('HOME')}/.cache/eeha_gui_cache"

def parseCSV(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = [header.strip() for header in next(csv_reader)]
        data_by_columns = {header: [] for header in headers}

        for row in csv_reader:
            for header, value in zip(headers, row):
                data_by_columns[header].append(value.strip())

        return data_by_columns
    
# Gets filtered data from YAML file
def getResultsYamlData(dataset):
    data = parseYaml(dataset['path'])
    data_filtered = {}
    try:
        ## FILTER UNUSED DATA TO AVOID MEMORY CONSUMPTION

        last_fit_tag = 'pr_data_' + str(data['pr_epoch'] - 1)
        last_val_tag = 'validation_' + str(data['val_epoch'] - 1)

        if last_val_tag not in data or last_fit_tag not in data:
            last_fit_tag = 'pr_data_' + str(data['train_data']['epoch_best_fit_index'])
            last_val_tag = 'validation_' + str(data['train_data']['epoch_best_fit_index'])

        data_filtered = {'validation_best': data[last_val_tag], 'pr_data_best': data[last_fit_tag],
                        'train_data': data['train_data'],'n_images': data['n_images'], 'pretrained': data['pretrained'],
                        'n_classes': data['dataset_info']['nc'], 'dataset_tag': data['dataset_tag']
                        }
    except KeyError as e:
        log(f"Missing key in results data dict({dataset['key']}): {e}", bcolors.ERROR)
        
    return data_filtered
                        

def getCSVData(dataset):
    csv_path = dataset['path'].replace('results.yaml','results.csv')
    if os.path.exists(csv_path):
        data= parseCSV(csv_path)
        return {'csv_data': data}
    else:
        log(f"\t· Could not parse CSV file associated to {dataset['key']}", bcolors.ERROR)
        return {}
    
def getArgsYamlData(dataset):
    arg_path = dataset['path'].replace('results.yaml','args.yaml')
    data_filtered = {}
    if os.path.exists(arg_path):
        try:
            data = parseYaml(arg_path)
            data_filtered = {'batch': data['batch'], 'deterministic': data['deterministic']}
        except KeyError as e:
            log(f"Missing key in Args data dict ({dataset['key']}): {e}", bcolors.ERROR)
    else:
        log(f"\t· Could not parse Args file file associated to {dataset['key']}", bcolors.ERROR)
    
    return data_filtered

# Wrap function to be paralelized
def background_load_data(dataset_key_tuple):
    key, dataset = dataset_key_tuple
    filename = f"{cache_path}/{key.replace('/','_')}.yaml.cache"
    
    if os.path.exists(filename):
        data = parseYaml(filename)
        # log(f"Loaded data from cache file in {filename}")
    else:
        data = getResultsYamlData(dataset)
        data.update(getCSVData(dataset))
        data.update(getArgsYamlData(dataset))
        log(f"Reloaded data from RAW data for {key} dataset")

    # log(f"\t· Parsed {dataset['key']} data")
    return data

def background_save_cache(dataset_key_tuple):
    key, dataset = dataset_key_tuple
    filename = f"{cache_path}/{key.replace('/','_')}.yaml.cache"
    # log(f"Data cache to be stored in {filename}")
    # if not os.path.exists(filename):
    dumpYaml(filename, dataset)
    # log(f"Stored data cache file in {filename}")

def find_results_file(search_path = test_path, file_name = data_file_name):
    log(f"Search all results.yaml files")

    dataset_info = {}
    for root, dirs, files in os.walk(search_path):
        if file_name in files:
            abs_path = os.path.join(root, file_name)
            if "validate" in abs_path: # Avoid validation tests, only training
                continue
            name = abs_path.split("/")[-3] + "/" + abs_path.split("/")[-2]
            dataset_info[name] = {'name': abs_path.split("/")[-2], 'path': abs_path, 'model': abs_path.split("/")[-3], 'key': name}

    # # Order dataset by name
    myKeys = list(dataset_info.keys())
    myKeys.sort()
    dataset_info = {i: dataset_info[i] for i in myKeys}
    return dataset_info
    
class DataSetHandler:
    def __init__(self, update_cache = True):
        self.update_cache = update_cache
        if update_cache and os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            log(f"Cleared previous cache files to be recached.")
        # Ensure cache dir exists if cleared or firs execution in machine or...
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self.dataset_info = find_results_file()
        self.parsed_data = {}

        # Load data in background
        self.lock = threading.Lock()
        self.futures_result = {}
        self.executor = ProcessPoolExecutor()
        self.futures = {key: self.executor.submit(background_load_data, (key,info)) for key, info in self.dataset_info.items()}

        self.monitor_data_load()
        
    def monitor_data_load(self):
        thread = threading.Thread(target=self.monitor_futures)
        thread.daemon = True
        thread.start()

    def monitor_futures(self):
        with self.lock:
            while self.futures:
                time.sleep(1)  # Esperar un segundo antes de comprobar de nuevo

                for key, future in list(self.futures.items()):
                    if future.done():
                        try:
                            self.futures_result[key] = future.result()
                        except Exception as e:
                            log(f"Exception catched processing future {key}: {e}", bcolors.ERROR)
                        finally:
                            del self.futures[key]
        log(f"All executors finished loading data. Store cache data")
        
        log(f"Update cache data files for later executions")
        self.executor.shutdown() # Clear previous executor
        self.executor = ProcessPoolExecutor()
        self.futures = {key: self.executor.submit(background_save_cache, (key, self.__getitem__(key))) for key in self.dataset_info.keys()}
        time.sleep(1)  # Esperar un segundo antes de comprobar de nuevo

        while self.futures:
            time.sleep(1)  # Esperar un segundo antes de comprobar de nuevo
            for key, future in list(self.futures.items()):
                if future.done():
                    del self.futures[key]
        log(f"Finished caching data")
            
    def getInfo(self):
        return self.dataset_info

    def keys(self):
        return self.dataset_info.keys()
    
    def __getitem__(self, key):
        if not key in self.parsed_data:
            with self.lock:
                self.parsed_data = self.futures_result

        return self.parsed_data[key]

    def __len__(self):
        return len(self.dataset_info)
    
    def __del__(self):
        # Destructor can be called before executed object is created if somth fails :(
        if hasattr(self, 'executor') and self.executor:
            # Cancelar todas las futuras que aún no han terminado
            for future in self.futures.values():
                future.cancel()

            # Apagar el pool de procesos
            self.executor.shutdown()
