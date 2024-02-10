
#!/usr/bin/env python3
# encoding: utf-8
"""
    Handles the loading of dataset data to be used later by the GUY
"""


import os
import sys

from datetime import datetime

from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import csv

from config_utils import yolo_output_path as test_path
from config_utils import log, bcolors, parseYaml

data_file_name = "results.yaml"

def parseCSV(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = [header.strip() for header in next(csv_reader)]
        data_by_columns = {header: [] for header in headers}

        for row in csv_reader:
            for header, value in zip(headers, row):
                data_by_columns[header].append(value.strip())

        return data_by_columns
    
# Wrap function to be paralelized
def background_load_data(dataset):
    data = parseYaml(dataset['path'])
    csv_path = dataset['path'].replace('results.yaml','results.csv')
    if os.path.exists(csv_path):
        data['csv_data'] = parseCSV(csv_path)
    else:
        log(f"\t· Could not parse CSV file associated to {dataset['key']}", bcolors.ERROR)
    log(f"\t· Parsed {dataset['key']} data")
    return data


class DataSetHandler:
    def __init__(self):
        self.dataset_info = self.find_results_file()
        self.parsed_data = {}

        # Load data in background
        self.executor = ProcessPoolExecutor()
        self.futures = {key: self.executor.submit(background_load_data, info) for key, info in self.dataset_info.items()}

    def find_results_file(self, search_path = test_path, file_name = data_file_name):
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

    def getInfo(self):
        return self.dataset_info

    def keys(self):
        return self.dataset_info.keys()
    
    def __getitem__(self, key):
        if not key in self.parsed_data:
            log(f"\t· Retrieve {key} data parsed from process executor")
            data = self.futures[key].result() 
            self.parsed_data[key] = data
            
            # remove this retrieved key from dic
            self.futures.pop(key)
            
            if not self.futures:
                # shutdown the process pool
                self.executor.shutdown() # blocks
        else:
            # log(f"\t· Already parsed {key} data")
            data = self.parsed_data[key]

        return data

    def __len__(self):
        return len(self.dataset_info)
    
    def __del__(self):
        if self.executor:
            # Cancelar todas las futuras que aún no han terminado
            for future in self.futures.values():
                future.cancel()

            # Apagar el pool de procesos
            self.executor.shutdown()