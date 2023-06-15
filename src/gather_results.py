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

from config import yolo_val_output as test_path

data_file_name = "results.yaml"

# CSV list of rows
row_list = [['Title', 'Date', 'Model', 'Dataset', 'Condition', 'Type', 'Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95']]

def parseYaml(file_path):
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)


def gatherCSVAllTests():
    for folder in os.listdir(test_path): # for each model named folder
        for sub_folder in os.listdir(f"{test_path}/{folder}"): # folder with each of the tests
            if data_file_name in os.listdir(f"{test_path}/{folder}/{sub_folder}"):
                path = os.path.join(test_path, folder, sub_folder, data_file_name)
                # print(f"File found in {path}")
                
                dataset_type = sub_folder.split("_")
                data_parsed = parseYaml(path)
                creation_date = datetime.fromtimestamp(os.path.getctime(path))
                for class_type, data in data_parsed['data'].items():
                    print(data)
                    # print(data)
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


def plot_curve(func):
    def wrapper_plot_curve(py, labels, save_dir, xlabel = 'Confidence', ylabel = 'Metric', *args, **kwargs):
        # Precision-recall curve
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        for py, color, label in zip(py, ('red', 'green', 'grey'), labels):
            for i, y in enumerate(py):
                func(*args, **kwargs, ax = ax, py = y, index = i, color = color, label = label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.set_title(f'{ylabel}-{xlabel} Curve')
        fig.savefig(save_dir, dpi=250)
        print(f"Stored new diagram in {save_dir}")
        plt.close(fig)
    return wrapper_plot_curve

@plot_curve
def plot_pr_curve(ax, px, py, ap, index, color, label, names=()):
    ax.plot(px, py, linewidth=1, label=f'{label} {names[index]} (ap = {ap[0][index]:.3f})', color=color)  # plot(recall, precision)

@plot_curve
def plot_mc_curve(ax, px, py, index, color, label, names=()):
    ax.plot(px, py, linewidth=1, label=f'{label} {names[index]}', color=color)  # plot(confidence, metric)

def plot_data(px, py, ap, f1, p, r, names, labels, path):
        plot_pr_curve(px = px,  py = py, ap = ap,
                      save_dir = f"{path}_combined_pr_curve.png",
                      names = names, xlabel = 'Recall', ylabel = 'Precision', labels = labels)
        plot_mc_curve(px = px, py = f1, save_dir = f'{path}_F1_curve.png', 
                      names = names, ylabel='F1', labels = labels)
        plot_mc_curve(px = px, py = p, save_dir = f'{path}_P_curve.png', 
                      names = names, ylabel='Precision', labels = labels)
        plot_mc_curve(px = px, py = r, save_dir = f'{path}_R_curve.png', 
                      names = names, ylabel='Recall', labels = labels)


def plotCombinedCurves():
    for folder in os.listdir(test_path): # for each model named folder
        # Filter only folders in given path
        performed_tests = [subfolder for subfolder in os.listdir(f"{test_path}/{folder}") if os.path.isdir(f"{test_path}/{folder}/{subfolder}")]
        # print(performed_tests)
        plot_pairs = {}
        for test in performed_tests:
            key = folder + "/" + test.split("_")[0]
            test = f"{test_path}/{folder}/{test}"
            plot_pairs[key] = ([test] + plot_pairs[key]) if key in plot_pairs else [test]

    for key, value in plot_pairs.items():
        if len(value) <2:
            print(f"[ERROR] Seems that {key} test has only one version performed: {value}")
            continue

        lwir_data = parseYaml(f"{value[0]}/{data_file_name}")
        visual_data = parseYaml(f"{value[1]}/{data_file_name}")

        path = f"{test_path}/{key}"
        labels = ("(lwir)", "(RGB)")
        plot_data(px = lwir_data['pr_data']['px'], 
                  py = (lwir_data['pr_data']['py'], visual_data['pr_data']['py']),
                  ap = lwir_data['pr_data']['ap'],
                  f1 = (lwir_data['pr_data']['f1'],  visual_data['pr_data']['f1']),
                  p = (lwir_data['pr_data']['p'],  visual_data['pr_data']['p']),
                  r = (lwir_data['pr_data']['r'], visual_data['pr_data']['r']),
                  names = lwir_data['pr_data']['names'],
                  labels = labels,
                  path = path)

if __name__ == '__main__':
    gatherCSVAllTests()   
    plotCombinedCurves()
    
    