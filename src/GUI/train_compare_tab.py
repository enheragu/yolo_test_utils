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
from GUI.check_box_widget import DatasetCheckBoxWidget
from GUI.figure_tab_widget import PlotTabWidget

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
        self.csv_tab = QWidget()
        self.figure_tab_widget.addTab(self.csv_tab, "Table")

        # Crear un QScrollArea dentro de la pestaña CSV
        csv_scroll_area = QScrollArea()
        csv_scroll_area.setWidgetResizable(True)
        self.csv_tab.layout = QVBoxLayout()
        self.csv_tab.layout.addWidget(csv_scroll_area)
        self.csv_tab.setLayout(self.csv_tab.layout)

        # Crear una tabla para mostrar datos CSV
        self.csv_table = QTableWidget()
        self.csv_table.setSortingEnabled(True)  # Habilitar la ordenación de columnas
        # self.csv_table.setDragDropMode(QTableWidget.InternalMove)  # Habilitar la reordenación de filas
        csv_scroll_area.setWidget(self.csv_table)

        self.csv_data = [] # raw list to be filled with the data which should be stored in csv file

    def save_plot(self):
        # Open a file dialog to select the saving location
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)", options=options)

        if file_name:
            self.figure_tab_widget.saveFigures(file_name)
          
            with open(f'{file_name}.csv', 'w', newline='') as file:
                log(f"Summary CVS data in stored {file_name}.csv")
                writer = csv.writer(file)
                writer.writerows(self.csv_data)

    def render_data(self):
        self.plot_p_r_f1_data()
        self.load_table_data()

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
                    last_fit_tag = 'pr_data_' + str(data['pr_epoch'] - 1)
                    last_val_tag = 'validation_' + str(data['val_epoch'] - 1)

                    best_epoch = str(data['train_data']['epoch_best_fit_index'] + 1)

                    px = data[last_fit_tag]['px']
                    py = [data[last_fit_tag][py_tag]]
                    names = data[last_fit_tag]['names']
                    model = data[last_val_tag]['model'].split("/")[-1]
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

    def load_table_data(self):
        # Limpiar la tabla antes de cargar nuevos datos

        row_list = [['Model', 'Condition', 'Type', 'P', 'R', 'Train Images', 'Val Images', 'Instances', 'mAP50', 'mAP50-95', 'Class', 'Dataset', 'Best epoch (index)', 'Train Duration (h)', 'Pretrained', 'Date', 'Title']]

        for key in self.dataset_checkboxes.getChecked():
            try:
                # Is done at the end of plotting, so data should already be in parsed dict             
                data = self.dataset_handler[key]

                val_tag = f"validation_{data['val_epoch']-1}"

                model = data[val_tag]['model'].split("/")[-1]
                dataset = data[val_tag]['name'].split("/")[-1]
                dataset_type = dataset.split("_")
                
                best_epoch = data['train_data']['epoch_best_fit_index']
                train_duration = f"{data['train_data']['train_duration_h']:.2f}"
                end_date_tag = data['train_data']['train_end_time']
                train_images = data['n_images']['train']
                val_images = data['n_images']['val']
                pretrained = data['pretrained']

                for class_type, data_class in data[val_tag]['data'].items():
                    if 'all' in class_type:
                        continue

                    # log(f"\t {data}")
                    date_tag = datetime.fromisoformat(end_date_tag).strftime('%Y-%m-%d_%H:%M:%S')
                    test_title = f'{model}_{dataset}_{date_tag}_{class_type}'
                    condition = 'night' if 'night' in dataset_type[0] else 'day'
                    row_list += [[model, condition, "_".join(dataset_type[1:]),
                                    "{:.4f}".format(data_class['P']), 
                                    "{:.4f}".format(data_class['R']), 
                                    train_images,
                                    val_images, 
                                    data_class['Instances'], 
                                    "{:.4f}".format(data_class['mAP50']), 
                                    "{:.4f}".format(data_class['mAP50-95']), 
                                    class_type, 
                                    dataset_type[0], 
                                    best_epoch,
                                    train_duration,
                                    pretrained,
                                    date_tag,
                                    test_title]]
            except KeyError as e:
                log(f"Key error problem generating CSV for {key}. Row wont be generated. Missing key in data dict: {e}", bcolors.ERROR)

        self.csv_table.clear()
        self.csv_table.setRowCount(0)

        # Crear una nueva tabla
        self.csv_table.setColumnCount(len(row_list[0]))
        self.csv_table.setHorizontalHeaderLabels(row_list[0])
        
        # Conectar el evento sectionClicked del encabezado de la tabla
        self.csv_table.horizontalHeader().sectionClicked.connect(self.sort_table)

        # Agregar las filas a la tabla
        for row_position, row_data in enumerate(row_list[1:]):
            # row_position = self.csv_table.rowCount()
            self.csv_table.insertRow(row_position)
            for col_position, col_value in enumerate(row_data):
                item = QTableWidgetItem(str(col_value))
                self.csv_table.setItem(row_position, col_position, item)

        
        self.csv_data = row_list
        # Actualizar la vista de la tabla
        self.csv_table.resizeColumnsToContents()
        self.csv_table.resizeRowsToContents()
        
        log(f"CSV data display completed")

    def sort_table(self, logical_index):
        # Manejar el evento de clic en el encabezado para ordenar la tabla
        self.csv_table.sortItems(logical_index)

