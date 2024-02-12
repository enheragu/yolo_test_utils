
#!/usr/bin/env python3
# encoding: utf-8
"""
    Handles the loading of dataset data to be used later by the GUY
"""


import os
import sys
import time

from datetime import datetime

import threading
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import csv

from config_utils import yolo_output_path as test_path
from config_utils import log, bcolors, parseYaml

data_file_name = "results.yaml"

## Manages in a separate thread the processes so that the dataloading 
## does not keep infinitely keeping those resources. Futures are deleted
## once they finish and a thread safe access is provided for result data
class FutureManager:
    def __init__(self, executor, futures):
        self.executor = executor
        self.futures = futures
        self.lock = threading.Lock()
        self.results = {}

    def monitor_futures(self):
        with self.lock:
            while self.futures:
                time.sleep(1)  # Esperar un segundo antes de comprobar de nuevo

                for key, future in list(self.futures.items()):
                    if future.done():
                        try:
                            self.results[key] = future.result()
                        except Exception as e:
                            log(f"Exception catched processing future {key}: {e}", bcolors.ERROR)
                        finally:
                            del self.futures[key]
        log(f"All executors finished loading data")

    def get_results(self):
        with self.lock:
            return self.results

    def start_monitoring(self):
        thread = threading.Thread(target=self.monitor_futures)
        thread.daemon = True
        thread.start()


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

        data_filtered = {'validation_best': data[last_val_tag], 'pr_data_best': data[last_fit_tag],
                        'train_data': data['train_data'],'n_images': data['n_images']
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
            data_filtered = {'batch': data['batch'], 'pretrained': data['pretrained'], 'deterministic': data['deterministic']}
        except KeyError as e:
            log(f"Missing key in Args data dict ({dataset['key']}): {e}", bcolors.ERROR)
    else:
        log(f"\t· Could not parse Args file file associated to {dataset['key']}", bcolors.ERROR)
    
    return data_filtered

# Wrap function to be paralelized
def background_load_data(dataset):
    data = getResultsYamlData(dataset)
    data.update(getCSVData(dataset))
    data.update(getArgsYamlData(dataset))

    # log(f"\t· Parsed {dataset['key']} data")
    return data

# Wrap function to be paralelized
## Threaded version
# def background_load_data(dataset):
#     combined_data = {}
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [
#             executor.submit(getResultsYamlData, dataset),
#             executor.submit(getCSVData, dataset),
#             executor.submit(getArgsYamlData, dataset)
#         ]
#         concurrent.futures.wait(futures)

#         for future in futures:
#             combined_data.update(future.result())
    
#     return combined_data

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
    def __init__(self):
        self.dataset_info = find_results_file()
        self.parsed_data = {}

        # Load data in background
        self.executor = ProcessPoolExecutor()
        self.futures = {key: self.executor.submit(background_load_data, info) for key, info in self.dataset_info.items()}

        self.future_manager = FutureManager(self.executor, self.futures)
        self.future_manager.start_monitoring()


    def getInfo(self):
        return self.dataset_info

    def keys(self):
        return self.dataset_info.keys()
    
    def __getitem__(self, key):
        if not key in self.parsed_data:
            self.parsed_data = self.future_manager.get_results()

        return self.parsed_data[key]

    def __len__(self):
        return len(self.dataset_info)
    
    def __del__(self):
        if self.executor:
            # Cancelar todas las futuras que aún no han terminado
            for future in self.futures.values():
                future.cancel()

            # Apagar el pool de procesos
            self.executor.shutdown()
