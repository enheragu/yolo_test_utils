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

import matplotlib.pyplot as plt
from scipy.stats import norm
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout, QWidget, QPushButton, QFileDialog, QGroupBox, QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from scipy.ndimage.filters import gaussian_filter1d

import mplcursors

from config_utils import log, bcolors, parseYaml
from GUI.base_tab import BaseClassPlotter
from GUI.dataset_manager import DataSetHandler
from GUI.Widgets.check_box_widget import DatasetCheckBoxWidget, GroupCheckBoxWidget
from GUI.Widgets.figure_tab_widget import PlotTabWidget
from GUI.Widgets.numeric_slider_input_widget import NumericSliderInputWidget
from GUI.Widgets.csv_table_widget import TrainCSVDataTable

tab_keys = ['PR Curve', 'P Curve', 'R Curve', 'F1 Curve', 'mAP50', 'mAP50-95']

class VarianceComparePlotter(BaseClassPlotter):
    def __init__(self, dataset_handler):
        super().__init__(dataset_handler, tab_keys)

        self.dataset_variance_checkboxes = GroupCheckBoxWidget(self.options_widget, dataset_handler, include = "variance_", exclude = None, title = f"Variance analysis sets:")
        self.options_layout.insertWidget(0, self.dataset_variance_checkboxes,1)

        self.dataset_train_checkboxes = DatasetCheckBoxWidget(self.options_widget, dataset_handler)
        self.options_layout.insertWidget(0, self.dataset_train_checkboxes,3)

        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.deselect_all_button.clicked.connect(lambda: (self.dataset_train_checkboxes.deselect_all(), self.dataset_variance_checkboxes.deselect_all()))

        self.plot_button = QPushButton(" Generate Plot ", self)
        self.plot_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_button.clicked.connect(self.render_data)

        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.save_button.clicked.connect(self.save_plot)

        self.buttons_layout.addWidget(self.deselect_all_button)
        self.buttons_layout.addWidget(self.plot_button)
        self.buttons_layout.addWidget(self.save_button)

        # Tab for CSV data
        self.csv_tab = TrainCSVDataTable(dataset_handler, [self.dataset_train_checkboxes,self.dataset_variance_checkboxes])
        self.figure_tab_widget.addTab(self.csv_tab, "Table")
    
    def save_plot(self):
        # Open a file dialog to select the saving location
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)", options=options)

        if file_name:
            self.figure_tab_widget.saveFigures(file_name)
            self.csv_tab.save_data(file_name)

    def render_data(self):
        self.plot_loss_metrics_data()
        self.csv_tab.load_table_data()

    def plot_loss_metrics_data(self):

        # PY is an interpolated versino to plot it with a consistent px value. That's why
        # it is used here instead of plotting raw p-r values averaged
        plot_data = {'PR Curve': {'py': 'py', 'xlabel': "Recall", "ylabel": 'Precision'},
                     'P Curve': {'py': 'p', 'xlabel': "Confidence", "ylabel": 'Precision'},
                     'R Curve': {'py': 'r', 'xlabel': "Confidence", "ylabel": 'Recall'},
                     'F1 Curve': {'py': 'f1', 'xlabel': "Confidence", "ylabel": 'F1'}}
        
        # Borrar gráficos previos
        self.figure_tab_widget.clear()

        # Plotear los datos de los datasets seleccionados
        # log(f"Parse YAML of selected datasets to plot, note that it can take some time:")
        for canvas_key in self.tab_keys:
            
            if canvas_key == 'mAP50' or canvas_key == 'mAP50-95':
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                # Limpiar el gráfico anterior
                # xlabel = canvas_key
                # ylabel = "Probability"
                

                ax = self.figure_tab_widget[canvas_key].add_axes([0.08, 0.08, 0.84, 0.86])
                for index, group in enumerate(self.dataset_variance_checkboxes.getChecked()):
                    keys = [key for key in self.dataset_handler.keys() if group in key]
                    data_y = np.array([])
                    for key in keys:
                        data = self.dataset_handler[key]
                        try:
                            data_y = np.append(data_y, data['validation_best']['data']['all'][canvas_key])
                        except KeyError as e:
                            log(f"[{self.__class__.__name__}] Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                    
                    # log(f"{group}: {len(data_y) = }")
                    label = group.replace("train_based_variance_", "").replace(".yaml", "")    
                    bin_size = 7
                    ax.hist(data_y, bins=bin_size, density=True, alpha=0.5, label=f'{label}; n = {len(data_y)}; bin = {bin_size}', color=colors[index], edgecolor=colors[index])
                    
                    mean = np.mean(data_y)
                    std = np.std(data_y)

                    x = np.linspace(mean-std*8, mean+std*8, 100)
                    y = norm.pdf(x, mean, std)
                    
                    for pos in np.arange(mean-std*2, mean+std*2, std):
                        ax.axvline(x=pos, color=colors[index], linewidth=1)

                    ax.plot(x, y, linestyle='--', linewidth=2, color=colors[index])
                
                ax.set_title(f'{xlabel} distribution')
            else:
                if canvas_key not in plot_data:
                    continue

                # Limpiar el gráfico anterior
                xlabel = plot_data[canvas_key]['xlabel']
                ylabel = plot_data[canvas_key]['ylabel']
                # log(f"Plotting {ylabel}-{x, 'mAP50', 'mAP50-95'label} Curve")

                # ax = self.figure_tab_widget[canvas_key].add_subplot(111) #add_axes([0.08, 0.08, 0.84, 0.86])
                ax = self.figure_tab_widget[canvas_key].add_axes([0.08, 0.08, 0.84, 0.86])
                py_tag = plot_data[canvas_key]['py']

                def getLastEpochData(key_data, raw_data_dict):
                    px = raw_data_dict['pr_data_best']['px']
                    py = raw_data_dict['pr_data_best'][key_data]
                    names = raw_data_dict['pr_data_best']['names']
                    model = raw_data_dict['validation_best']['model'].split("/")[-1]
                    return px,py,names,model
                
                ## Plot each training result
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for index, key in enumerate(self.dataset_train_checkboxes.getChecked()):
                    data = self.dataset_handler[key]
                    try:
                        best_epoch = str(data['train_data']['epoch_best_fit_index'] + 1)
                        px, py, names, model = getLastEpochData(py_tag, data)
                        
                        for i, y in enumerate(py):
                            index_max = len(y)-1
                            # Filter by max recall values so to not have interpolated diagram
                            if canvas_key == 'PR Curve':
                                r = data['pr_data_best']['r'][i]
                                max_r = max(r)
                                index_max = next((i for i, x in enumerate(px) if x > max_r), None)
                            ax.plot(px[:index_max], y[:index_max], linewidth=2, label=f"{self.dataset_handler.getInfo()[key]['name']} ({model}) {names[i]} (best epoch: {best_epoch})")  # plot(confidence, metric)

                    except KeyError as e:
                        log(f"[{self.__class__.__name__}] Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)

                ## Plot each variance group
                # number_std = self.std_plot_widget.value
                prev_idx = len(self.dataset_train_checkboxes.getChecked())
                for index, group in enumerate(self.dataset_variance_checkboxes.getChecked()):
                    keys = [key for key in self.dataset_handler.keys() if group in key]
                    py_vec = []
                    for key in keys:
                        data = self.dataset_handler[key]
                        try:
                            px, py, names, model = getLastEpochData(py_tag, data)
                            if len(names) > 1:
                                log(f"[{self.__class__.__name__}] -----------------------------------------------------------------------", bcolors.ERROR)
                                log(f"[{self.__class__.__name__}] - (TDB) MORE THAN ONE TARGET CLASS. NOT BEING HANDLED CORRECTLY (TBD) -", bcolors.ERROR)
                                log(f"[{self.__class__.__name__}] -----------------------------------------------------------------------", bcolors.ERROR)
                            for i, y in enumerate(py):
                                # Filter by max recall values so to not have interpolated diagram. Gets the max index possible
                                # between all recall curves involved in mean and max/min
                                index_max = len(y)-1
                                if canvas_key == 'PR Curve':
                                    r = data['pr_data_best']['r'][i]
                                    max_r = max(r)
                                    index_max = next((i for i, x in enumerate(px) if x > max_r), None)
                                
                                # FIll from index_max to the end with NAN data
                                y = y[:index_max] + [np.nan] * (len(y) - index_max)
                                py_vec.append(y) #[:index_max])

                        except KeyError as e:
                            log(f"[{self.__class__.__name__}] Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)

                    if not py_vec:
                        log(f"[{self.__class__.__name__}] Vector of PY data empty. {key} variance curve wont be plotted.", bcolors.WARNING)
                        continue

                    data_matrix = np.array(py_vec)
                    # nanmean make average ignoring NAN stuff
                    mean = np.nanmean(data_matrix, axis = 0)
                    max_vals = np.nanmax(data_matrix, axis=0)
                    min_vals = np.nanmin(data_matrix, axis=0)

                    # std = np.std(data_matrix, axis = 0, ddof=1)
                    # conf_interval = std*number_std
                    # conf_interval = 1.96*std/math.sqrt(len(keys))
                    # conf_interval = Z(2 STD) * Error típico (la desviación típica que tendría el estadístico si lo calcularas en infinitas muestras iguales)
                    label = group.replace("train_based_variance_", "").replace(".yaml", "")
                                
                    ax.plot(px, mean, label=f"{label}; n = {len(py_vec)}", color=colors[index+prev_idx])
                    ax.fill_between(px, min_vals, max_vals, alpha=0.3, facecolor=colors[index+prev_idx])

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                # self.ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
                ax.set_title(f'{ylabel}-{xlabel} Curve')

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
            
            # Configurar leyenda
            ax.legend()

            # Use a Cursor to interactively display the label for a selected line.
            self.cursor[canvas_key] = mplcursors.cursor(ax, hover=True)
            self.cursor[canvas_key].connect("add", lambda sel, xlabel=xlabel, ylabel=ylabel: sel.annotation.set(
                text=f"{sel.artist.get_label().split(' ')[0]}\n{xlabel}: {sel.target[0]:.2f}, {ylabel}: {sel.target[1]:.2f}",
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgrey', alpha=0.7)
            ))

        # Actualizar los gráfico
        self.figure_tab_widget.draw()

        log(f"[{self.__class__.__name__}] Parsing and plot PR-P-R-F1 graphs finished")
