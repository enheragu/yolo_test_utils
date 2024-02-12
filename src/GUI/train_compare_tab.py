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

import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout, QWidget, QPushButton, QCheckBox, QFileDialog, QGroupBox, QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from scipy.ndimage.filters import gaussian_filter1d

import mplcursors

from config_utils import log, bcolors, parseYaml
from GUI.dataset_manager import DataSetHandler
from GUI.Widgets.check_box_widget import DatasetCheckBoxWidget
from GUI.Widgets.figure_tab_widget import PlotTabWidget
from GUI.Widgets.csv_table_widget import TrainCSVDataTable

class TrainComparePlotter(QScrollArea):
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

        self.dataset_checkboxes = DatasetCheckBoxWidget(self.options_widget, dataset_handler)
        self.options_layout.addWidget(self.dataset_checkboxes,3)

        # Crear un widget que contendrá los grupos los botones
        buttons_widget = QWidget(self.options_widget)
        self.options_layout.addWidget(buttons_widget,1)
        buttons_layout = QGridLayout(buttons_widget)

        ## Create a button to select all checkboxes from a given condition
        self.select_all_day_button = QPushButton(" Select 'day' ", self)
        self.select_all_day_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.select_all_day_button.clicked.connect(lambda: self.dataset_checkboxes.select_cond('day'))
        self.select_all_night_button = QPushButton(" Select 'night' ", self)
        self.select_all_night_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.select_all_night_button.clicked.connect(lambda: self.dataset_checkboxes.select_cond('night'))
        self.select_all_all_button = QPushButton(" Select 'day-night' ", self)
        self.select_all_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.select_all_all_button.clicked.connect(lambda: self.dataset_checkboxes.select_cond('all'))

        self.select_all_button = QPushButton(" Select All ", self)
        self.select_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.select_all_button.clicked.connect(self.dataset_checkboxes.select_all)

        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.deselect_all_button.clicked.connect(self.dataset_checkboxes.deselect_all)

        self.plot_button = QPushButton(" Generate Plot ", self)
        self.plot_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_button.clicked.connect(self.render_data)

        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.save_button.clicked.connect(self.save_plot)

        buttons_layout.addWidget(self.select_all_day_button, 0, 0, 1, 1)
        buttons_layout.addWidget(self.select_all_night_button, 0, 1, 1, 1)
        buttons_layout.addWidget(self.select_all_all_button, 1, 0, 1, 1)
        buttons_layout.addWidget(self.select_all_button, 1, 1, 1, 1)
        buttons_layout.addWidget(self.deselect_all_button, 0, 2, 2, 1)
        buttons_layout.addWidget(self.plot_button, 2, 0, 1, 3)
        buttons_layout.addWidget(self.save_button, 3, 0, 1, 3)

        self.cursor = {}
        self.tab_keys = ['PR Curve', 'P Curve', 'R Curve', 'F1 Curve']
        self.figure_tab_widget = PlotTabWidget(self.tab_keys)        
        self.layout.addWidget(self.figure_tab_widget)
        
        # Tab for CSV data
        self.csv_tab = TrainCSVDataTable(dataset_handler, self.dataset_checkboxes)
        self.figure_tab_widget.addTab(self.csv_tab, "Table")

    def save_plot(self):
        # Open a file dialog to select the saving location
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)", options=options)

        if file_name:
            self.figure_tab_widget.saveFigures(file_name)
            self.csv_tab.save_data(file_name)

    def render_data(self):
        self.plot_p_r_f1_data()
        self.csv_tab.load_table_data()

    # Plots PR, P, R and F1 curve from each dataset involved
    def plot_p_r_f1_data(self):

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
            for key in self.dataset_checkboxes.getChecked():
                data = self.dataset_handler[key]
                py_tag = plot_data[canvas_key]['py']

                try:
                    best_epoch = str(data['train_data']['epoch_best_fit_index'] + 1)

                    px = data['pr_data_best']['px']
                    py = [data['pr_data_best'][py_tag]]
                    names = data['pr_data_best']['names']
                    model = data['validation_best']['model'].split("/")[-1]
                    for py_list in py:
                        for i, y in enumerate(py_list):
                            ax.plot(px, y, linewidth=2, label=f"{self.dataset_handler.getInfo()[key]['name']} ({model}) {names[i]} (best epoch: {best_epoch})")  # plot(confidence, metric)

                except KeyError as e:
                    log(f"Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)

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