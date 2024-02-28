
#!/usr/bin/env python3
# encoding: utf-8
"""
    Handles the loading of dataset data to be used later by the GUY
"""


import os
import sys
import time
import shutil
import inspect

from datetime import datetime

import threading
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import csv

from config_utils import yolo_output_path as test_path
from config_utils import parseYaml, dumpYaml
from log_utils import log, bcolors

data_file_name = "results.yaml"
ignore_file_name = "EEHA_GUI_IGNORE" # If a file is found in path with this name the folder would be ignored
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
                        'n_classes': data['dataset_info']['nc'], 'dataset_tag': data['dataset_tag'],
                        'device_type': data['system_data']['device_type']
                        }
    except KeyError as e:
        log(f"[{inspect.currentframe().f_code.co_name}] Missing key in results data dict({dataset['key']}): {e}", bcolors.ERROR)
        
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
            log(f"[{inspect.currentframe().f_code.co_name}] Missing key in Args data dict ({dataset['key']}): {e}", bcolors.ERROR)
    else:
        log(f"\t· Could not parse Args file file associated to {dataset['key']}", bcolors.ERROR)
    
    return data_filtered

# Wrap function to be paralelized
def background_load_data(dataset_key_tuple):
    key, dataset, update_cache = dataset_key_tuple
    filename = f"{cache_path}/{key.replace('/','_')}.yaml.cache"
    
    if os.path.exists(filename) and not update_cache:
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

"""
    :param: ignored ignore or not the folders with ignore files. False to process all
        even with ignore file
"""
def find_results_file(search_path = test_path, file_name = data_file_name, ignored = True):
    log(f"Search all results.yaml files")

    dataset_info = {}
    for root, dirs, files in os.walk(search_path):

        if ignore_file_name in files and ignored:
            log(f"Path with {ignore_file_name} file is set to be ignored: {root}.", bcolors.WARNING)
            dirs[:] = [] # Clear subdir list to avoid getting into them
            continue

        if file_name in files:
            abs_path = os.path.join(root, file_name)
            if "validate" in abs_path: # Avoid validation tests, only training
                log(f"Validate tests are to be ignored: {abs_path}.", bcolors.WARNING)
                continue
            name = abs_path.split("/")[-3] + "/" + abs_path.split("/")[-2]
            dataset_info[name] = {'name': abs_path.split("/")[-2], 'path': abs_path, 'model': abs_path.split("/")[-3], 'key': name}

    # # Order dataset by name
    myKeys = list(dataset_info.keys())
    myKeys.sort()
    dataset_info = {i: dataset_info[i] for i in myKeys}
    return dataset_info
    
class DataSetHandler:
    def __init__(self, update_cache = True, search_path = test_path):
        self.new(update_cache, search_path)

    def new(self, update_cache = True, search_path = test_path):
        global cache_path

        # Prepares different cache path for Dataset Handler from different location than default
        if search_path != test_path:
            cache_path = cache_path + "_extra"
            log(f"Loading data from different directory: {search_path}")
            log(f"Redirecting cache to {cache_path}")

        self.update_cache = update_cache
        if update_cache and os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            log(f"Cleared previous cache files to be recached.")
        # Ensure cache dir exists if cleared or firs execution in machine or...
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self.dataset_info = find_results_file(search_path)
        self.parsed_data = {}
        self.incomplete_dataset = {}

        self.access_data_lock = threading.Lock()
        self.save_data_lock = threading.Lock()

        self._load_data(self.dataset_info, self.update_cache)
        
    def _load_data(self, dataset_dict, update_cache):
        # Load data in background
        self.futures_result = {}
        self.executor_load = ProcessPoolExecutor()
        self.futures_load = {key: self.executor_load.submit(background_load_data, (key,info,update_cache)) for key, info in dataset_dict.items()}
        
        thread = threading.Thread(target=self._monitor_futures_load)
        thread.daemon = True
        thread.start()

        thread_save = threading.Thread(target=self._save_cache_data, args=(dataset_dict,))
        thread_save.daemon = True
        thread_save.start()
              
    def _save_cache_data(self, dataset_dict):
        with self.save_data_lock:
            log(f"Update cache data files for later executions")
            self.executor_save = ProcessPoolExecutor()
            self.futures_save = {key: self.executor_save.submit(background_save_cache, (key, self.__getitem__(key))) for key in dataset_dict.keys()}
            
            thread = threading.Thread(target=self._monitor_futures_save)
            thread.daemon = True
            thread.start()

    def _monitor_futures_load(self):
        with self.save_data_lock:
            with self.access_data_lock:
                while self.futures_load:
                    time.sleep(1)  # Esperar un segundo antes de comprobar de nuevo

                    for key, future in list(self.futures_load.items()):
                        if future.done():
                            try:
                                self.futures_result[key] = future.result()
                            except Exception as e:
                                log(f"Exception catched processing future {key}: {e}", bcolors.ERROR)
                            finally:
                                del self.futures_load[key]
            self.executor_load.shutdown() # Clear executor
            log(f"All executors finished loading data.")
        
        
    def _monitor_futures_save(self):
        time.sleep(1)
        while self.futures_save:
            time.sleep(1)  # Esperar un segundo antes de comprobar de nuevo
            for key, future in list(self.futures_save.items()):
                if future.done():
                    del self.futures_save[key]
        self.executor_save.shutdown() # Clear executor
        log(f"All executors finished caching data")

    def reloadIncomplete(self):
        log(f"Reload incomplete datasets: {self.incomplete_dataset.keys()}")
        if self.incomplete_dataset:
            self._load_data(self.incomplete_dataset, update_cache=True)

    def getInfo(self):
        return self.dataset_info

    def keys(self):
        return self.dataset_info.keys()
    
    # Remove dataset from execution as it is giving problems
    def markAsIncomplete(self, key):
        log(f"[{self.__class__.__name__}] Incomplete dataset data {key}. Won't be taken into account.", bcolors.WARNING)
        incomplete_data = self.__delitem__(key)
        self.incomplete_dataset[key] = incomplete_data

    def __delitem__(self, key):
        if key in self.parsed_data:
            self.dataset_info.pop(key)
            eliminate = self.parsed_data.pop(key)
            return eliminate
        else:
            raise KeyError(f'[{self.__class__.__name__}] Key {key} is not in parsed_data dict.')
        
    def __getitem__(self, key):
        if key in self.incomplete_dataset:
            return None
        
        if not key in self.parsed_data:
            with self.access_data_lock:
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
