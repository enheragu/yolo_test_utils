#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt tab view with all plot available to compare between different training runs
"""

import os
import sys

from datetime import datetime


import csv
import math

import numpy as np
from PyQt6.QtWidgets import QGridLayout, QWidget, QPushButton, QFileDialog, QSizePolicy

import mplcursors
import matplotlib.pyplot as plt
import seaborn as sns

from utils import parseYaml
from utils import log, bcolors
from GUI.base_tab import BaseClassPlotter
from GUI.Widgets import DatasetCheckBoxWidget, TrainCSVDataTable, DialogWithCheckbox

tab_keys = ['PR Curve', 'P Curve', 'R Curve', 'F1 Curve', 'MR Curve',
             'mAP50', 'mAP50-95', 'recall', 'F1', 'MissRate', 'FFPI']
equations = {
        'P': r'$P(c) = \dfrac{TP(c)}{TP(c) + FP(c)}$',
        'R': r'$R(c) = \dfrac{TP(c)}{TP(c) + FN(c)}$',
        'MR': r'M$R(c) = \dfrac{FN(c)}{TP(c) + FN(c)} = 1 - Recall(c)$',
        'F1': r'F$1(c) = \dfrac{TP(c)}{TP(c) + \dfrac{1}{2}*(FP(c) + FN(c))}$',
        'PR': r'$P(R)$'
}

class TrainComparePlotter(BaseClassPlotter):
    def __init__(self, dataset_handler):
        super().__init__(dataset_handler, tab_keys)

        # Needs to overload buttons widget with Grid layout for this specific view
        self.options_layout.removeWidget(self.buttons_widget)
        self.buttons_widget = QWidget(self.options_widget)
        self.options_layout.addWidget(self.buttons_widget,1)
        self.buttons_layout = QGridLayout(self.buttons_widget)

        self.dataset_checkboxes = DatasetCheckBoxWidget(self.options_widget, dataset_handler, title_filter=["train_based_"])
        self.options_layout.insertWidget(0, self.dataset_checkboxes,3)

        ## Create a button to select all checkboxes from a given condition
        self.select_all_day_button = QPushButton(" Select 'day' ", self)
        self.select_all_day_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.select_all_day_button.clicked.connect(lambda: self.dataset_checkboxes.select_cond('day'))
        self.select_all_night_button = QPushButton(" Select 'night' ", self)
        self.select_all_night_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.select_all_night_button.clicked.connect(lambda: self.dataset_checkboxes.select_cond('night'))
        self.select_all_all_button = QPushButton(" Select 'day-night' ", self)
        self.select_all_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.select_all_all_button.clicked.connect(lambda: self.dataset_checkboxes.select_cond('all'))

        ## --- Adds window selector to be able to add manually individual tests from variance_ stuff
        self.dataset_checkboxes_extra = DatasetCheckBoxWidget(self.options_widget, dataset_handler, exclude = None, include="variance_", title_filter=["train_based_"], max_rows = 8)
        self.dataset_checkboxes_extra.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.select_extra_button = QPushButton(" Select extra ")
        self.select_extra_button.setToolTip('Allows to choose single variance tests instead of plotting them as a group')
        self.select_extra_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.extra_dataset_dialog = DialogWithCheckbox(title="Extra dataset selector", checkbox_widget=self.dataset_checkboxes_extra, render_func = self.render_data)
        self.select_extra_button.clicked.connect(self.extra_dataset_dialog.show)
        ## ---

        self.select_all_button = QPushButton(" Select All ", self)
        self.select_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.select_all_button.clicked.connect(self.dataset_checkboxes.select_all)

        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.deselect_all_button.clicked.connect(lambda: (self.dataset_checkboxes.deselect_all(), self.dataset_checkboxes_extra.deselect_all()))

        self.plot_button = QPushButton(" Generate Plot ", self)
        self.plot_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_button.clicked.connect(self.render_data)

        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.save_button.clicked.connect(self.save_plot)

        self.buttons_layout.addWidget(self.select_all_day_button, 0, 0, 1, 1)
        self.buttons_layout.addWidget(self.select_all_night_button, 0, 1, 1, 1)
        self.buttons_layout.addWidget(self.select_all_all_button, 1, 0, 1, 1)
        self.buttons_layout.addWidget(self.select_all_button, 1, 1, 1, 1)
        self.buttons_layout.addWidget(self.deselect_all_button, 0, 2, 2, 1)
        self.buttons_layout.addWidget(self.plot_button, 2, 0, 1, 3)
        self.buttons_layout.addWidget(self.save_button, 3, 0, 1, 3)
        self.buttons_layout.addWidget(self.select_extra_button, 4, 0, 1, 3)
        
        if tab_keys:
            self.change_labels_button = QPushButton(" Edit labels ", self)
            self.change_labels_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.change_labels_button.clicked.connect(self.figure_tab_widget.edit_labels)
            self.buttons_layout.addWidget(self.change_labels_button, 5,0,1,3)
       
        # Tab for CSV data
        self.csv_tab = TrainCSVDataTable(dataset_handler, [self.dataset_checkboxes, self.dataset_checkboxes_extra])
        self.figure_tab_widget.addTab(self.csv_tab, "Table")
    
    def update_checkbox(self):
        self.dataset_checkboxes.update_checkboxes()
        self.dataset_checkboxes_extra.update_checkboxes()

    def save_plot(self):
        # Open a file dialog to select the saving location
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)")

        if file_name:
            self.figure_tab_widget.saveFigures(file_name)
            self.csv_tab.save_data(file_name)

    def render_data(self):

        checked = self.dataset_checkboxes_extra.getChecked() + self.dataset_checkboxes.getChecked()
        self.plot_p_r_f1_data(checked)
        self.csv_tab.load_table_data()
        log(f"[{self.__class__.__name__}] Plot and table updated")

    # Plots PR, P, R and F1 curve from each dataset involved
    def plot_p_r_f1_data(self, checked_list):
        # PY is an interpolated versino to plot it with a consistent px value<
        plot_data = {'PR Curve': {'py': 'py', 'xlabel': "Recall", "ylabel": 'Precision'},
                     'P Curve': {'py': 'p', 'xlabel': "Confidence", "ylabel": 'Precision'},
                     'R Curve': {'py': 'r', 'xlabel': "Confidence", "ylabel": 'Recall'},
                     'F1 Curve': {'py': 'f1', 'xlabel': "Confidence", "ylabel": 'F1'},
                     'MR Curve': {'py': 'mr_plot', 'xlabel': "Confidence", "ylabel": 'Miss Rate'},
                     'mAP50-95': {'py': 'mAP50-95', 'xlabel': "Test", "ylabel": 'mAP50-95'},
                     'mAP50': {'py': 'mAP50', 'xlabel': "Test", "ylabel": 'mAP50'},
                     'recall': {'py': 'R', 'xlabel': "Test", "ylabel": 'Recall'},
                     'F1': {'py': 'f1', 'xlabel': "Test", "ylabel": 'F1'},
                     'MissRate': {'py': 'mr', 'xlabel': "Test", "ylabel": 'MissRate'},
                     'FFPI': {'py': 'ffpi', 'xlabel': "Test", "ylabel": 'FFPI'}}
        
        # Borrar gráficos previos
        self.figure_tab_widget.clear()

        # Plotear los datos de los datasets seleccionados
        # log(f"Parse YAML of selected datasets to plot, note that it can take some time:")
        for canvas_key in self.tab_keys:
            
            # Limpiar el gráfico anterior
            xlabel = plot_data[canvas_key]['xlabel']
            ylabel = plot_data[canvas_key]['ylabel']
            # log(f"Plotting {ylabel}-{xlabel} Curve")

            ax = self.figure_tab_widget[canvas_key].add_subplot(111) #add_axes([0.08, 0.08, 0.84, 0.86])
            # ax = self.figure_tab_widget[canvas_key].add_axes([0.1, 0.08, 0.84, 0.86])

            if canvas_key in ['mAP50', 'mAP50-95', 'recall'] \
             or canvas_key in ['F1','MissRate', 'FFPI']:
                labels = {}
                values = {}
                for key in checked_list:
                    data = self.dataset_handler[key]
                    if not data:
                        continue
                    
                    try:
                        group_name = data['validation_best']['name'].split('/')[0]
                        if not group_name in labels:
                            labels[group_name] = []
                            values[group_name] = []

                        if canvas_key in ['F1', 'MissRate', 'FFPI']:
                            data_tag = plot_data[canvas_key]['py']
                            y_data = data['pr_data_best'].get(f"m{data_tag}", data['pr_data_best'].get(data_tag))
                            if y_data is None:
                                print(f'{y_data = }; {canvas_key = }')
                            values[group_name].append(y_data[0])
                            labels[group_name].append(self.dataset_handler.getInfo()[key]['label'])
                        else:
                            data_tag = plot_data[canvas_key]['py']
                            values[group_name].append(data['validation_best']['data']['all'].get(f"m{data_tag}", data['validation_best']['data']['all'].get(data_tag)))
                            labels[group_name].append(self.dataset_handler.getInfo()[key]['label'])
                    
                    except KeyError as e:
                        log(f"[{self.__class__.__name__}] Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                        self.dataset_handler.markAsIncomplete(key)
                    except TypeError as e:
                        log(f"[{self.__class__.__name__}] Key error problem generating curve for {key} for {py_tag} plot. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                    
                for (group, values), labels in zip(values.items(), labels.values()):
                    # sorted_pairs = sorted(zip(values, labels))
                    # sorted_values, sorted_labels = zip(*sorted_pairs)
                    # sns.scatterplot(x=sorted_labels, y=sorted_values, ax = ax, label=f'{group}')
                    sns.scatterplot(x=labels, y=values, ax = ax, label=f'{group}')
                    # ax.set_xticks(np.arange(len(labels)))
                    # ax.set_xticklabels(labels, rotation=45, ha='right')
                
                # for i, y in enumerate(sorted_values):
                #     ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
                #     ax.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
                
                labels = ax.get_xticklabels()
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=20, ha='right')

                ax.grid(True, linestyle='--', linewidth=0.5)
                ax.set_title(f'{ylabel} comparison')

                # Use a Cursor to interactively display the label for a selected line.
                self.cursor[canvas_key] = mplcursors.cursor(ax, hover=True)
                self.cursor[canvas_key].connect("add", lambda sel, xlabel=xlabel, ylabel=ylabel: sel.annotation.set(
                    text=f"{sel.artist.get_label().split(' ')[0:2]}\n{xlabel}: {ax.get_xticklabels()[int(sel.index)].get_text()}, {ylabel}: {sel.target[1]:.2f}",
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgrey', alpha=0.7)
                ))

            elif canvas_key == 'MR curve':
                ax.text(0.5,0.5, 'Disabled for now', ha='center', va='center', fontsize=28, color='gray')

            else:
                for key in checked_list:
                    data = self.dataset_handler[key]
                    if not data:
                        continue
                    
                    py_tag = plot_data[canvas_key]['py']

                    try:
                        best_epoch = str(data['train_data']['epoch_best_fit_index'] + 1)
                        px = data['pr_data_best'].get('px_plot', data['pr_data_best'].get('px'))
                        py = data['pr_data_best'].get(f'{py_tag}_plot', data['pr_data_best'].get(py_tag))
                            
                        names = data['pr_data_best']['names']
                        model = data['validation_best']['model'].split("/")[-1]
                        for i, y in enumerate(py):
                            # Filter by max recall values so to not have interpolated diagram
                            if canvas_key == 'PR Curve':
                                r_list = data['pr_data_best'].get('r_plot', data['pr_data_best'].get('r'))
                                r = r_list[i]
                                max_r = max(r)
                                index_max = next((i for i, x in enumerate(px) if x > max_r), None)
                                y = y[:index_max] + [np.nan] * (len(y) - index_max)
                            
                            # ax.plot(px, y, linewidth=2, label=f"{self.dataset_handler.getInfo()[key]['name']} ({model}) {names[i]} (best epoch: {best_epoch})")  # plot(confidence, metric)
                            ax_label = self.dataset_handler.getInfo()[key]['label']
                            sns.lineplot(x=px, y=y, linewidth=2, label=ax_label, ax = ax)

                            

                    except KeyError as e:
                        log(f"[{self.__class__.__name__}] Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                        self.dataset_handler.markAsIncomplete(key)
                    except TypeError as e:
                        log(f"[{self.__class__.__name__}] Key error problem generating curve for {key} for {py_tag} plot. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                     
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                # Configurar leyenda
                # self.ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
                ax.legend()
                ax.set_title(f'{ylabel}-{xlabel} Curve')

                background_eq_tag = canvas_key.replace(" Curve", "")
                if self.plot_background_img and background_eq_tag in equations:

                    eq = equations[background_eq_tag]
                    x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) * 0.5
                    y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) * 0.5
                    ax.text(x_center, y_center, eq, color="Black", alpha=0.5, fontsize=16, ha="center", va="center")

                else:
                    print(f"{background_eq_tag} not set.")

                # Use a Cursor to interactively display the label for a selected line.
                self.cursor[canvas_key] = mplcursors.cursor(ax, hover=True)
                self.cursor[canvas_key].connect("add", lambda sel, xlabel=xlabel, ylabel=ylabel: sel.annotation.set(
                    text=f"{' '.join(sel.artist.get_label().split(' ')[0:2])}\n{xlabel}: {sel.target[0]:.2f}, {ylabel}: {sel.target[1]:.2f}",
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgrey', alpha=0.7)
                ))

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            self.figure_tab_widget[canvas_key].ax.append(ax)
                
        # Actualizar los gráfico
        self.figure_tab_widget.draw()

        # log(f"[{self.__class__.__name__}] Parsing and plot PR-P-R-F1 graphs finished")