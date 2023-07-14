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

from config_utils import yolo_output_path as test_path

data_file_name = "results.yaml"

# CSV list of rows
row_list = [['Title', 'Date', 'Model', 'Dataset', 'Condition', 'Type', 'Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95']]

def parseYaml(file_path):
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)


def gatherCSVAllTests():
    global row_list
    for folder in os.listdir(test_path): # for each model named folder
        if not os.path.isdir(f"{test_path}/{folder}"):
            continue

        for sub_folder in os.listdir(f"{test_path}/{folder}"): # folder with each of the tests
            if not os.path.isdir(f"{test_path}/{folder}/{sub_folder}"):
                continue

            if data_file_name in os.listdir(f"{test_path}/{folder}/{sub_folder}"):
                path = os.path.join(test_path, folder, sub_folder, data_file_name)
                # print(f"File found in {path}")
                
                dataset_type = sub_folder.split("_")
                data_parsed = parseYaml(path)
                creation_date = datetime.fromtimestamp(os.path.getctime(path))
                for class_type, data in data_parsed['data'].items():
                    print(data)
                    model = folder#.replace(".pt", "")
                    date_tag = creation_date.strftime('%Y-%m-%d_%H:%M:%S')
                    test_title = f'{model}_{sub_folder}_{date_tag}_{class_type}'
                    condition = 'night' if 'night' in dataset_type[0] else 'day'
                    row_list += [[test_title, creation_date, model, dataset_type[0], condition, dataset_type[1], 
                                    class_type, data['Images'], data['Instances'], 
                                    "{:.4f}".format(data['P']), 
                                    "{:.4f}".format(data['R']), 
                                    "{:.4f}".format(data['mAP50']), 
                                    "{:.4f}".format(data['mAP50-95'])]]
                print(f"Found {data_file_name} in {folder = }; {sub_folder = }")

    with open('summary_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


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
#         print(f"Stored new diagram in {save_dir}")
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
#     # print(f"Plot {save_dir}")
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
#     print(f"Stored new diagram in {save_dir}")
#     plt.close(fig)



def plot_curve(px, py, names = [], ap = [], labels = [], save_dir = "", title_name = "", xlabel = 'Confidence', ylabel = 'Metric', *args, **kwargs):
    # Precision-recall curve
    ap_labels = ap
    if ap_labels == []:
        ap_labels = [""]*len(labels)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for py_list, color, label, ap_iter in zip(py, ('red', 'green', 'grey', 'blue'), labels, ap_labels):
        for i, y in enumerate(py_list):
            ap_text = f" (ap = {ap_iter[0][i]:.3f})" if ap_iter != "" else ""
            ax.plot(px, y, linewidth=1, label=f'{label} {names[i]}', color=color)  # plot(confidence, metric)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{title_name} {ylabel}-{xlabel} Curve')
    fig.savefig(save_dir, dpi=250)
    print(f"Stored new diagram in {save_dir}")
    plt.close(fig)

def plot_data(px, py, ap, f1, p, r, names, labels, path, title_name = ""):
    plot_curve(px = px,  py = py, ap = ap,
                    save_dir = f"{path}_combined_pr_curve.png",
                    names = names, title_name = title_name, xlabel = 'Recall', ylabel = 'Precision', labels = labels)
    plot_curve(px = px, py = f1, save_dir = f'{path}_F1_curve.png', 
                    names = names, title_name = title_name, ylabel='F1', labels = labels)
    plot_curve(px = px, py = p, save_dir = f'{path}_P_curve.png', 
                    names = names, title_name = title_name, ylabel='Precision', labels = labels)
    plot_curve(px = px, py = r, save_dir = f'{path}_R_curve.png', 
                    names = names, title_name = title_name, ylabel='Recall', labels = labels)


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
            # Ensure same order so graphis use same colours
            plot_pairs[key].sort()

    print(f"Plot pairs are:")
    for key, value_list in plot_pairs.items():
        print(f"\t· {key}:")
        for value in value_list:
            print(f"\t\t - {value}")
    print()

    for key, value in plot_pairs.items():
        if len(value) <2:
            print(f"[ERROR] Seems that {key} test has only one version performed: {value}")
            continue
        
        # Get all dataset yaml configuration
        data = []
        for test in value:
            data += [parseYaml(f"{test}/{data_file_name}")]
            
        path = f"{test_path}/{key}"
        plot_data(px = data[0]['pr_data']['px'], 
                  py = [test['pr_data']['py'] for test in data],
                  ap = [test['pr_data']['ap'] for test in data],
                  f1 = [test['pr_data']['f1'] for test in data],
                  p = [test['pr_data']['p'] for test in data],
                  r = [test['pr_data']['r'] for test in data],
                  names = data[0]['pr_data']['names'],
                  labels = [test['test'].split("/")[-1].split("_")[-1] for test in data],
                  path = path,
                  title_name = f"{key}  -  ")

if __name__ == '__main__':
    print(f"Process results from {test_path}")
    gatherCSVAllTests()   
    plotCombinedCurves()
    
    