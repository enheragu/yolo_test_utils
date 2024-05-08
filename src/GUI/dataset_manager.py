
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
import re

from datetime import datetime

import threading
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import traceback

import csv

if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    
from argument_parser import yolo_output_path
from utils import parseYaml, dumpYaml
from utils import log, bcolors

data_file_name = "results.yaml"
ignore_file_name = "EEHA_GUI_IGNORE" # If a file is found in path with this name the folder would be ignored
cache_path = f"{os.getenv('HOME')}/.cache/eeha_gui_cache"
cache_extension = '.yaml.cache'
test_key_clean = ['_4090', '_3090', '_GPU3090', '_A30', r'_[0-9]{8}'] # Path tags to be cleared from key (merges tests from different GPUs). Leave empty for no merging

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
    if not data:
        log(f"[{inspect.currentframe().f_code.co_name}] Empty data in {dataset['key']}).", bcolors.ERROR)
        return {}

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
    key, dataset, update_cache, load_from_cache = dataset_key_tuple
    filename = f"{cache_path}/{key}{cache_extension}"
    
    if (os.path.exists(filename) and not update_cache) or load_from_cache:
        data = parseYaml(filename)
        # log(f"Loaded data from cache file in {filename}")

    else:
        data = getResultsYamlData(dataset)
        data.update(getCSVData(dataset))
        data.update(getArgsYamlData(dataset))
        log(f"Reloaded data from RAW data for {key} dataset")

    # log(f"\t· Parsed {dataset['key']} data")
    ## Update cache data from data currently parsed
    cache_key_path = f'{cache_path}/{key.split("/")[0]}'
    os.makedirs(cache_key_path, exist_ok=True)
    dumpYaml(filename, data)
    
    return data

"""
    :param: ignored ignore or not the folders with ignore files. False to process all
        even with ignore file
"""
def find_results_file(search_path = yolo_output_path, file_name = data_file_name, ignored = True):
    global test_key_clean
    log(f"Search all {file_name} files in {search_path}")

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
            name = abs_path.split("/")[-2]
            title = abs_path.split("/")[-2]
            model = abs_path.split("/")[-3]
            for clear_pattern in test_key_clean:
                # model = model.replace(clear_tag, "")
                model = re.sub(clear_pattern, "", model)
                title = re.sub(clear_pattern, "", title)
            key = f"{model}/{name}"
            dataset_info[key] = {'name': name, 'path': abs_path, 'model': model, 'key': key, 'title': f"{title}"}

    ## Order dataset by name
    myKeys = list(dataset_info.keys())
    myKeys.sort()
    dataset_info = {i: dataset_info[i] for i in myKeys}
    return dataset_info

"""
    Equivalent to find_results_file when data is to be loaded directly from cache
"""
def find_cache_file(search_path = cache_path, file_name = cache_extension):
    global test_key_clean
    log(f"Search all {file_name} files in {search_path}")

    dataset_info = {}
    for root, dirs, files in os.walk(search_path):
        
        for file in files:
            if file_name in file:
                abs_path = os.path.join(root, file)
                key_name = abs_path.replace(file_name, "")
                name = key_name.split("/")[-1]
                title = abs_path.split("/")[-1]
                model = key_name.split("/")[-2]
                for clear_pattern in test_key_clean:
                    # model = model.replace(clear_tag, "")
                    model = re.sub(clear_pattern, "", model)
                    title = re.sub(clear_pattern, "", title)
                key = f"{model}/{name}"
                dataset_info[key] = {'name': name, 'path': abs_path, 'model': model, 'key': key, 'title': f"{title}"}

    myKeys = list(dataset_info.keys())
    myKeys.sort()
    dataset_info = {i: dataset_info[i] for i in myKeys}
    return dataset_info

class DataSetHandler:
    def __init__(self, update_cache = True, search_path = yolo_output_path):
        self.new(update_cache, search_path)

    def new(self, update_cache = True, search_path = yolo_output_path, load_from_cache = False):
        global cache_path
        self.update_cache = update_cache
        self.load_from_cache = load_from_cache
        
        if load_from_cache:
            self.dataset_info = find_cache_file()
        else:
            # Prepares different cache path for Dataset Handler from different location than default
            if search_path != yolo_output_path:
                cache_path = cache_path + "_extra"
                log(f"Loading data from different directory: {search_path}")
                log(f"Redirecting cache to {cache_path}")

            if update_cache and os.path.exists(cache_path):
                shutil.rmtree(cache_path)
                log(f"Cleared previous cache files to be recached.")
            # Ensure cache dir exists if cleared or firs execution in machine or...
            os.makedirs(cache_path, exist_ok=True)

            self.dataset_info = find_results_file(search_path)

        self.parsed_data = {}
        self.incomplete_dataset = {}

        self.access_data_lock = threading.Lock()

        self._load_data(self.dataset_info)
        
    def _load_data(self, dataset_dict, update_cache = None):
        if update_cache:
            self.update_cache = update_cache

        # Load data in background
        self.futures_result = {}
        self.executor_load = ProcessPoolExecutor()
        self.futures_load = {key: self.executor_load.submit(background_load_data, (key,info,self.update_cache, self.load_from_cache)) for key, info in dataset_dict.items()}
        
        thread = threading.Thread(target=self._monitor_futures_load)
        thread.daemon = True
        thread.start()

    def _monitor_futures_load(self):
            with self.access_data_lock:
                while self.futures_load:
                    time.sleep(1)  # Esperar un segundo antes de comprobar de nuevo

                    for key, future in list(self.futures_load.items()):
                        if future.done():
                            try:
                                self.futures_result[key] = future.result()
                            except Exception as e:
                                log(f"Exception catched processing future {key}: {e}", bcolors.ERROR)
                                log(traceback.format_exc(), bcolors.ERROR)
                            finally:
                                del self.futures_load[key]
            self.executor_load.shutdown() # Clear executor
            os.system('notify-send "GUI - Dataset manager" "Data loading is completed"')
            os.system('paplay /usr/share/sounds/freedesktop/stereo/complete.oga')
            log(f"All executors finished loading data.")
        

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
        incomplete_data = self.__delitem__(key)
        log(f"[{self.__class__.__name__}] Incomplete dataset data {key}. Won't be taken into account. Test data in: {incomplete_data['path'].replace('results.yaml', '')}", bcolors.WARNING)
        self.incomplete_dataset[key] = incomplete_data

    def __delitem__(self, key):
        if key in self.parsed_data:
            dataset_info = self.dataset_info.pop(key)
            eliminate = self.parsed_data.pop(key)
            return dataset_info
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



"""
    Main execution allows to cache data ind advance without loading the whole gui :)
"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GUI to review training results.")
    parser.add_argument('--update_cache', action='store_true', help='Actualizar archivos de caché si es verdadero')

    # Parsear los argumentos de la línea de comandos
    args = parser.parse_args()

    update_cache = args.update_cache
    dataset_handler = DataSetHandler(update_cache)