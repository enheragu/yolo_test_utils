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
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout, QWidget, QPushButton, QFileDialog, QGroupBox, QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from scipy.ndimage.filters import gaussian_filter1d

import mplcursors

from config_utils import log, bcolors, parseYaml
from GUI.dataset_manager import DataSetHandler
from GUI.Widgets.check_box_widget import DatasetCheckBoxWidget, GroupCheckBoxWidget
from GUI.Widgets.figure_tab_widget import PlotTabWidget
from GUI.Widgets.numeric_slider_input_widget import NumericSliderInputWidget

class VarianceComparePlotter(QScrollArea):
    def __init__(self, dataset_handler):
        super().__init__()

        self.dataset_handler = dataset_handler

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.setWidgetResizable(True)  # Permitir que el widget se expanda
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.options_widget = QWidget(self)
        self.layout.addWidget(self.options_widget)

        self.options_layout = QHBoxLayout()
        self.options_widget.setLayout(self.options_layout)
        self.options_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.dataset_train_checkboxes = DatasetCheckBoxWidget(self.options_widget, dataset_handler)
        self.options_layout.addWidget(self.dataset_train_checkboxes,3)

        self.dataset_variance_checkboxes = GroupCheckBoxWidget(self.options_widget, dataset_handler, include = "variance_", exclude = None, title = f"Variance analysis sets:")
        self.options_layout.addWidget(self.dataset_variance_checkboxes,1)

        # Crear un widget que contendrá los grupos los botones
        buttons_widget = QWidget(self.options_widget)
        self.options_layout.addWidget(buttons_widget,1)
        buttons_layout = QVBoxLayout(buttons_widget)

        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.deselect_all_button.clicked.connect(lambda: (self.dataset_train_checkboxes.deselect_all(), self.dataset_variance_checkboxes()))

        self.plot_button = QPushButton(" Generate Plot ", self)
        self.plot_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_button.clicked.connect(self.render_data)

        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.save_button.clicked.connect(self.save_plot)

        # self.std_plot_widget = NumericSliderInputWidget('STD plot:')
        # self.deselect_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # buttons_layout.addWidget(self.std_plot_widget)

        buttons_layout.addWidget(self.deselect_all_button)
        buttons_layout.addWidget(self.plot_button)
        buttons_layout.addWidget(self.save_button)

        self.tab_keys = ['PR Curve', 'P Curve', 'R Curve', 'F1 Curve']
        self.figure_tab_widget = PlotTabWidget(self.tab_keys)        
        self.layout.addWidget(self.figure_tab_widget)
        
        self.cursor = {}
        
    def save_plot(self):
        # Open a file dialog to select the saving location
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)", options=options)

        if file_name:
            self.figure_tab_widget.saveFigures(file_name)

    def render_data(self):
        self.plot_loss_metrics_data()

    def plot_loss_metrics_data(self):

        plot_data = {'PR Curve': {'py': 'py', 'xlabel': "Recall", "ylabel": 'Precision'},
                     'P Curve': {'py': 'p', 'xlabel': "Confidence", "ylabel": 'Precision'},
                     'R Curve': {'py': 'r', 'xlabel': "Confidence", "ylabel": 'Recall'},
                     'F1 Curve': {'py': 'f1', 'xlabel': "Confidence", "ylabel": 'F1'}}
        
        # Borrar gráficos previos
        self.figure_tab_widget.clear()

        # Plotear los datos de los datasets seleccionados
        log(f"Parse YAML of selected datasets to plot, note that it can take some time:")
        for canvas_key in self.tab_keys:
            if canvas_key not in plot_data:
                continue

            # Limpiar el gráfico anterior
            xlabel = plot_data[canvas_key]['xlabel']
            ylabel = plot_data[canvas_key]['ylabel']
            log(f"Plotting {ylabel}-{xlabel} Curve")

            # ax = self.figure_tab_widget[canvas_key].add_subplot(111) #add_axes([0.08, 0.08, 0.84, 0.86])
            ax = self.figure_tab_widget[canvas_key].add_axes([0.08, 0.08, 0.84, 0.86])
            py_tag = plot_data[canvas_key]['py']

            def getLastEpochData(key_data, raw_data_dict):
                px = raw_data_dict['pr_data_best']['px']
                py = [raw_data_dict['pr_data_best'][key_data]]
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
                    for py_list in py:
                        for i, y in enumerate(py_list):
                            ax.plot(px, y, linewidth=2, label=f"{self.dataset_handler.getInfo()[key]['name']} ({model}) {names[i]} (best epoch: {best_epoch})", color=colors[index])

                except KeyError as e:
                    log(f"Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)

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
                            log(f"-----------------------------------------------------------------------", bcolors.ERROR)
                            log(f"- (TDB) MORE THAN ONE TARGET CLASS. NOT BEING HANDLED CORRECTLY (TBD) -", bcolors.ERROR)
                            log(f"-----------------------------------------------------------------------", bcolors.ERROR)
                        for py_list in py:
                            for i, y in enumerate(py_list):
                                py_vec.append(y)

                    except KeyError as e:
                        log(f"Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)

                if not py_vec:
                    log(f"Vector of PY data empty. {key} variance curve wont be plotted.", bcolors.WARNING)
                    continue

                data_matrix = np.array(py_vec)
                mean = np.mean(data_matrix, axis = 0)
                max_vals = np.max(data_matrix, axis=0)
                min_vals = np.min(data_matrix, axis=0)

                # std = np.std(data_matrix, axis = 0, ddof=1)
                # conf_interval = std*number_std
                # conf_interval = 1.96*std/math.sqrt(len(keys))
                # conf_interval = Z(2 STD) * Error típico (la desviación típica que tendría el estadístico si lo calcularas en infinitas muestras iguales)
                label = group.replace("train_based_variance_", "").replace(".yaml", "")
                ax.plot(px, mean, label=f"{label}; n = {len(keys)}", color=colors[index+prev_idx])
                ax.fill_between(px, min_vals, max_vals, alpha=0.3, facecolor=colors[index+prev_idx])

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            # self.ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
            ax.set_title(f'{ylabel}-{xlabel} Curve')
            
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

        log(f"Parsing and plot PR-P-R-F1 graphs finished")
