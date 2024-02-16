#!/usr/bin/env python3
# encoding: utf-8

"""
    Gathers all the results from executions of yolo made with simple_test script 
    in a table
"""

import os
import yaml
from yaml.loader import SafeLoader
from datetime import datetime
from pathlib import Path
import numpy as np

import csv
import matplotlib.pyplot as plt

from multiprocessing.pool import Pool, ThreadPool
from functools import partial

from config_utils import yolo_output_path as test_path
from config_utils import log, bcolors, parseYaml

## CONFIGURATION
gather_csv_data = False
plot_pr = True
plot_f1 = False
plot_p = False
plot_r = False


data_file_name = "results.yaml"

# CSV list of rows
row_list = [['Title', 'Model', 'Condition', 'Type', 'P', 'R', 'Images', 'Instances', 'mAP50', 'mAP50-95', 'Class', 'Dataset', 'Date']]

# Split the data processin into a separate function to apply multiprocessing
def gatherData(data): 
    folder, sub_folder, path = data
    row_list = []

    dataset_type = sub_folder.split("_")
    data_parsed = parseYaml(path)
    creation_date = datetime.fromtimestamp(os.path.getctime(path))
    for class_type, data in data_parsed['data'].items():
        # log(f"\t {data}")
        model = folder#.replace(".pt", "")
        date_tag = creation_date.strftime('%Y-%m-%d_%H:%M:%S')
        test_title = f'{model}_{sub_folder}_{date_tag}_{class_type}'
        condition = 'night' if 'night' in dataset_type[0] else 'day'
        row_list += [[test_title, model, condition, dataset_type[1], 
                        "{:.4f}".format(data['P']), 
                        "{:.4f}".format(data['R']), 
                        data['Images'], 
                        data['Instances'], 
                        "{:.4f}".format(data['mAP50']), 
                        "{:.4f}".format(data['mAP50-95']), 
                        class_type, 
                        dataset_type[0], 
                        creation_date]]
            
    return row_list
    
def gatherCSVAllTests():
    global row_list
    data = []
    log(f"Check for {data_file_name} to gather data:")
    for folder in os.listdir(test_path): # for each model named folder
        if not os.path.isdir(f"{test_path}/{folder}"):
            continue

        for sub_folder in os.listdir(f"{test_path}/{folder}"): # folder with each of the tests
            if not os.path.isdir(f"{test_path}/{folder}/{sub_folder}"):
                continue

            if data_file_name in os.listdir(f"{test_path}/{folder}/{sub_folder}"):
                data += [[folder, sub_folder, os.path.join(test_path, folder, sub_folder, data_file_name)]]
                log(f"\t· Found {data_file_name} in {folder}/{sub_folder}")
    
    log(f"Parse and gather all data into a single list")
    with Pool() as pool:
        for result in pool.map(gatherData, data):
            row_list += result

    with open(f'{test_path}/summary_data.csv', 'w', newline='') as file:
        log(f"Summary CVS data in stored {test_path}/summary_data.csv")
        writer = csv.writer(file)
        writer.writerows(row_list)

"""
# def plot_curve(func):
#     def wrapper_plot_curve(py, labels, save_dir, title_name = "", xlabel = 'Confidence', ylabel = 'Metric', *args, **kwargs):
#         # Precision-recall curve
#         fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
#         for py, color, label in zip(py, ('red', 'green', 'grey', 'blue'), labels):
#             for i, y in enumerate(py):
#                 func(*args, **kwargs, ax = ax, py = y, index = i, color = color, label = label)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
#         ax.set_title(f'{title_name} {ylabel}-{xlabel} Curve')
#         fig.savefig(save_dir, dpi=250)
#         log(f"Stored new diagram in {save_dir}")
#         plt.close(fig)
#     return wrapper_plot_curve

# @plot_curve
# def plot_pr_curve(ax, px, py, ap, index, color, label, names=()):
#     # TBD handle ap list correctly
#     ax.plot(px, py, linewidth=1, label=f'{label} {names[index]} (ap = {ap[0][0][index]:.3f})', color=color)  # plot(recall, precision)

# @plot_curve
# def plot_mc_curve(ax, px, py, index, color, label, names=()):
#     ax.plot(px, py, linewidth=1, label=f'{label} {names[index]}', color=color)  # plot(confidence, metric)


# def plot_curve(px, py, names, ap = [], labels = [], save_dir = "", title_name = "", xlabel = 'Confidence', ylabel = 'Metric'):
#     # Precision-recall curve
#     # log(f"Plot {save_dir}")
#     ap_labels = ap
#     if not ap_labels:
#         ap_labels = ""*len(labels)

#     fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    
#     for py, color, label, ap, name in zip(py, ('red', 'green', 'grey', 'blue'), labels, ap_labels, names):
#         for i, y in enumerate(py):
#             ap_text = f" (ap = {ap[0][i]:.3f})" if ap != "" else ""
#             ax.plot(px, y, linewidth=1, label=f'{label} {name}{ap_text}', color=color)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
#     ax.set_title(f'{title_name} {ylabel}-{xlabel} Curve')
#     fig.savefig(save_dir, dpi=250)
#     log(f"Stored new diagram in {save_dir}")
#     plt.close(fig)
"""


def plot_curve(px, py, names = [], ap = [], labels = [], save_dir = "", title_name = "", xlabel = 'Confidence', ylabel = 'Metric', *args, **kwargs):
    # Precision-recall curve
    ap_labels = ap
    if ap_labels == []:
        ap_labels = [""]*len(labels)
    
    # with plt.xkcd():
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for py_list, label, ap_iter in zip(py, labels, ap_labels):
        for i, y in enumerate(py_list):
            ap_text = f" (ap = {ap_iter[0][i]:.3f})" if ap_iter != "" else ""
            ax.plot(px, y, linewidth=1, label=f'{label} {names[i]}')  # plot(confidence, metric)
            # ax.scatter(px, y, marker='x', label=f'{label} {names[i]}')  # plot(confidence, metric)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.legend()
    ax.set_title(f'{title_name} {ylabel}-{xlabel} Curve')
    fig.savefig(save_dir, dpi=450)
    log(f"Stored new diagram in {save_dir}")
    plt.close(fig)

def plot_data_graphs(px, py, ap, f1, p, r, names, labels, path, title_name = ""):
    if plot_pr:
        plot_curve(px = px,  py = py, ap = ap,
                        save_dir = f"{path}_combined_pr_curve.png",
                        names = names, title_name = title_name, xlabel = 'Recall', ylabel = 'Precision', labels = labels)
    if plot_f1:
        plot_curve(px = px, py = f1, save_dir = f'{path}_F1_curve.png', 
                        names = names, title_name = title_name, ylabel='F1', labels = labels)
    if plot_p: 
        plot_curve(px = px, py = p, save_dir = f'{path}_P_curve.png', 
                        names = names, title_name = title_name, ylabel='Precision', labels = labels)
    if plot_r:
        plot_curve(px = px, py = r, save_dir = f'{path}_R_curve.png', 
                        names = names, title_name = title_name, ylabel='Recall', labels = labels)

# Encapsulate in separate function to make it paralel
def plotData(plot_pair):
    key, value = plot_pair
    if len(value) <2:
        log(f"[WARNING] Seems that {key} test has only one version performed: {value}", bcolors.WARNING)
        return
    
    # Get all dataset yaml configuration
    data = []
    yaml_files = [f"{test}/{data_file_name}" for test in value]
    with ThreadPool() as pool:
        for result in pool.map(parseYaml, yaml_files):
            data += [result]
    
    path = f"{test_path}/{key}"
    plot_data_graphs(px = data[0]['pr_data']['px'], 
                py = [test['pr_data']['py'] for test in data],
                ap = [test['pr_data']['ap'] for test in data],
                f1 = [test['pr_data']['f1'] for test in data],
                p = [test['pr_data']['p'] for test in data],
                r = [test['pr_data']['r'] for test in data],
                names = data[0]['pr_data']['names'],
                labels = ["_".join(test['test'].split("/")[-1].split("_")[1:]) + " (" + test['model'].split("/")[-1] + ")" for test in data],
                path = path,
                title_name = f"{key}  -  ")

def plotCombinedCurves():
    plot_pairs = {}
    for folder in os.listdir(test_path): # for each model named folder
        if not os.path.isdir(f"{test_path}/{folder}"):
            continue
    
        # Filter only folders in given path
        performed_tests = [subfolder for subfolder in os.listdir(f"{test_path}/{folder}") if os.path.isdir(f"{test_path}/{folder}/{subfolder}") and data_file_name in os.listdir(f"{test_path}/{folder}/{subfolder}")]
        
        for test in performed_tests:
            key = folder + "/" + test.split("_")[0]
            test = f"{test_path}/{folder}/{test}"
            plot_pairs[key] = ([test] + plot_pairs[key]) if key in plot_pairs else [test]
            # Ensure same order so graphics use same colours
            plot_pairs[key].sort()

    log(f"Plot pairs are:")
    for key, value_list in plot_pairs.items():
        log(f"\t· {key}:")
        for value in value_list:
            log(f"\t\t - {value}")


    # Iterate image generation multiprocessing
    with Pool() as pool:
        pool.map(plotData, list(plot_pairs.items()))        

if __name__ == '__main__':
    log(f"Process results from {test_path}")
    log(f"{gather_csv_data = }; {plot_pr = }; {plot_f1 = }; {plot_p = }; {plot_r = };")
    if gather_csv_data:
        gatherCSVAllTests()   
    if plot_pr or plot_f1 or plot_p or plot_r:
        plotCombinedCurves()
    
    