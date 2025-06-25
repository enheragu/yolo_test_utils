#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt widget to display data from trainig into a CSV table format
"""
import os
import itertools
from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QScrollArea, QSizePolicy, QVBoxLayout, QTableWidget, QTableWidgetItem

from utils import log, bcolors
from utils.log_utils import logTable
from utils.log_utils import printDictKeys
from .check_box_widget import DatasetCheckBoxWidget, GroupCheckBoxWidget, BestGroupCheckBoxWidget

TABLE_DECIMAL_PRECISION="{:.5f}"

class TrainCSVDataTable(QWidget):
    """
        :param: dataset_handler instance of shared DataSetHandler
        :param: dataset_checkboxes is a check_box_widget or a list of them (any of both types)
    """
    def __init__(self, dataset_handler, dataset_checkboxes):
        super().__init__()

        self.dataset_handler = dataset_handler
        self.dataset_checkboxes = dataset_checkboxes  if isinstance(dataset_checkboxes, list) else [dataset_checkboxes]
        

        # self.setWidgetResizable(True)  # Permitir que el widget se expanda
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Crear un QScrollArea dentro de la pestaña CSV
        csv_scroll_area = QScrollArea()
        csv_scroll_area.setWidgetResizable(True)
        self.layout = QVBoxLayout()
        self.layout.addWidget(csv_scroll_area)
        self.setLayout(self.layout)

        # Crear una tabla para mostrar datos CSV
        self.csv_table = QTableWidget()
        self.csv_table.setSortingEnabled(True)  # Habilitar la ordenación de columnas
        # self.csv_table.setDragDropMode(QTableWidget.InternalMove)  # Habilitar la reordenación de filas
        csv_scroll_area.setWidget(self.csv_table)

        self.csv_data = [] # raw list to be filled with the data which should be stored in csv file

    def save_data(self,file_name = None):
        if file_name:          
            with open(f'{file_name}.csv', 'w', newline='') as file:
                log(f"[{self.__class__.__name__}] Summary CVS data in stored {file_name}.csv")
                writer = csv.writer(file)
                writer.writerows(self.csv_data)

            logTable(self.csv_data, os.path.join(file_name.split('/')[:-1]), file_name.split('/')[-1])

    ## Overloadable function :)
    def getDataDictToPlot(self):
        data = {}
        
        for checkbox_list in self.dataset_checkboxes:
            if isinstance(checkbox_list, DatasetCheckBoxWidget):
                for key in checkbox_list.getChecked():           
                    data[key] = self.dataset_handler[key]
            elif isinstance(checkbox_list, GroupCheckBoxWidget):
                for group in checkbox_list.getChecked():
                    keys = [key for key in self.dataset_handler.keys() if group in key]
                    for key in keys:
                        data[key] = self.dataset_handler[key]
            else:
                log(f"[{self.__class__.__name__}] CheckBox object instanced not recogniced.", bcolors.ERROR)
        return data
    
    def getDataDictToPlotAveraged(self):
        data = []
        for checkbox_list in self.dataset_checkboxes:
            if isinstance(checkbox_list, DatasetCheckBoxWidget):
                pass
            elif isinstance(checkbox_list, BestGroupCheckBoxWidget):
                pass
            elif isinstance(checkbox_list, GroupCheckBoxWidget):
                data += checkbox_list.getChecked()  
            else:
                log(f"[{self.__class__.__name__}] CheckBox object instanced not recogniced.", bcolors.ERROR)
        return data

    def load_table_data(self):
        # Limpiar la tabla antes de cargar nuevos datos

        row_list = [['Model', 'Condition', 'Type', 'P', 'R', 'mAP50', 'mAP50-95', 'MR', 'LAMR', 'FPPI',
                     'Class', 'Dataset', 'Best epoch (index)', 'Train Duration (h)', 
                     'Pretrained', 'Deterministic', 'Batch Size', 'Train Img', 'Val Img', 
                     'Instances', 'Num Classes', 'Dataset', 'Device', 'Date', 'Title', 'Group Key']]
        row_list_averaged = [[]] # Empty line, no title needed here

        for key, data in self.getDataDictToPlot().items():
            if not data:
                continue
            try:
                model = data['validation_best']['model'].split("/")[-1]
                dataset = data['validation_best']['name'].split("/")[-1]
                dataset_type = dataset.split("_")
                
                end_date_tag = data['train_data']['train_end_time']
                for class_type, data_class in data['validation_best']['data'].items():
                    if 'all' in class_type:
                        continue

                    # log(f"\t {data}")
                    date_tag = datetime.fromisoformat(end_date_tag).strftime('%Y-%m-%d_%H:%M:%S')
                    test_title = f'{model}_{dataset}_{date_tag}_{class_type}'
                    if 'night' in dataset_type[0]:
                        condition = 'night'
                    elif 'day' in dataset_type[0]:
                        condition = 'day'
                    elif 'all' in dataset_type[0]:
                        condition = 'all'
                    else:
                        condition = "Unknown"
                    
                    # printDictKeys(data)
                    row_list.append([model, condition, dataset_type[1],
                                    TABLE_DECIMAL_PRECISION.format(data_class.get('P', data_class.get(f"mP"))), 
                                    TABLE_DECIMAL_PRECISION.format(data_class.get('R', data_class.get(f"mR"))), 
                                    TABLE_DECIMAL_PRECISION.format(data_class['mAP50']), 
                                    TABLE_DECIMAL_PRECISION.format(data_class['mAP50-95']), 
                                    TABLE_DECIMAL_PRECISION.format(data['pr_data_best']['mr'][0]),
                                    TABLE_DECIMAL_PRECISION.format(data['pr_data_best']['lamr'][0]),
                                    TABLE_DECIMAL_PRECISION.format(data['pr_data_best']['fppi'][0]),
                                    class_type, 
                                    dataset_type[0], 
                                    data['train_data']['epoch_best_fit_index'],
                                    f"{data['train_data']['train_duration_h']:.2f}",
                                    data['pretrained'],
                                    data['deterministic'],
                                    data['batch'],
                                    int(data['n_images']['train']),
                                    int(data['n_images']['val']), 
                                    data_class['Instances'], 
                                    data['n_classes'],
                                    data['dataset_tag'],
                                    data['device_type'],
                                    date_tag,
                                    test_title,
                                    key.split("/")[0]])
            except KeyError as e:
                log(f"[{self.__class__.__name__}] Key error problem generating CSV for {key}. Row wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                self.dataset_handler.markAsIncomplete(key)

        for group in self.getDataDictToPlotAveraged():
            if not "variance" in group:
                continue
            keys = [key for key in self.dataset_handler.keys() if group in key]
            bestfit_epoch_vec = []
            train_duration_vec = []
            p_vec_vec = []
            r_vec_vec = []
            mAP50_vec = []
            mAP50_95_vec = []
            instances_vec = []
            mr_vec = []
            lamr_vec = []
            fppi_vec = []
            for key in keys:
                data = self.dataset_handler[key]
                if not data:
                    continue
                try:
                    bestfit_epoch_vec.append(data['train_data']['epoch_best_fit_index'])
                    train_duration_vec.append(data['train_data']['train_duration_h'])
                    p_vec_vec.append(data_class.get('P', data_class.get(f"mP")))
                    r_vec_vec.append(data_class.get('R', data_class.get(f"mR")))
                    mAP50_vec.append(data_class['mAP50'])
                    mAP50_95_vec.append(data_class['mAP50-95'])
                    mr_vec.append(data['pr_data_best']['mr'][0])
                    lamr_vec.append(data['pr_data_best']['lamr'][0])
                    fppi_vec.append(data['pr_data_best']['fppi'][0])
                    for class_type, data_class in data['validation_best']['data'].items():
                        if 'all' in class_type:
                            continue
                        instances_vec.append(data_class['Instances'])

                except KeyError as e:
                    log(f"[{self.__class__.__name__}] Key error problem generating averaged CSV for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                    self.dataset_handler.markAsIncomplete(key)

            model = data['validation_best']['model'].split("/")[-1]
            dataset = data['validation_best']['name'].split("/")[-1]
            dataset_type = dataset.split("_")

            for class_type, data_class in data['validation_best']['data'].items():
                if 'all' in class_type:
                    continue

                date_tag = "-"
                test_title = f'{model}_{dataset}_{class_type}'
                if 'night' in dataset_type[0]:
                    condition = 'night'
                elif 'day' in dataset_type[0]:
                    condition = 'day'
                elif 'all' in dataset_type[0]:
                    condition = 'all'
                else:
                    condition = "Unknown"

                for tag, function in [("mean", np.mean), ("std", np.std)]:
                    row_list_averaged.append([model, condition, dataset_type[1],
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(p_vec_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(r_vec_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(mAP50_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(mAP50_95_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(mr_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(lamr_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(fppi_vec, axis = 0))}", 
                                class_type, 
                                dataset_type[0], 
                                f"({tag}) {function(bestfit_epoch_vec, axis = 0):.2f}",
                                f"({tag}) {function(train_duration_vec, axis = 0):.2f}",
                                data['pretrained'],
                                data['deterministic'],
                                data['batch'],
                                int(data['n_images']['train']),
                                int(data['n_images']['val']), 
                                f"({tag}) {function(instances_vec, axis = 0):.2f}", 
                                data['n_classes'],
                                data['dataset_tag'],
                                data['device_type'],
                                date_tag,
                                f"{test_title}_{tag}",
                                f"{keys[0].split('/')[0]}/{tag}"])
                

        self.csv_table.clear()
        self.csv_table.setRowCount(0)

        # Crear una nueva tabla
        self.csv_table.setColumnCount(len(row_list[0]))
        self.csv_table.setHorizontalHeaderLabels(row_list[0])
        
        # Conectar el evento sectionClicked del encabezado de la tabla
        self.csv_table.horizontalHeader().sectionClicked.connect(self.sort_table)

        # Agregar las filas a la tabla
        row_color = {}
        def addRowsColored(row_list, alpha_value = 20):
            colors_list = sns.color_palette()
            color_iterator = itertools.cycle(colors_list)
            for row_position, row_data in enumerate(row_list[1:]):
                # row_position = self.csv_table.rowCount()
                self.csv_table.insertRow(row_position)
                
                ## Add colors
                ## row_tag Filters by group/condition_type
                row_tag = row_data[-1].split("/")[0]+"/"+row_data[1]
                if row_tag in row_color:
                    cell_color = row_color[row_tag]
                else:
                    next_color = next(color_iterator)
                    if isinstance(next_color, str):
                        next_color = to_rgba(next_color)
                    cell_color = QColor(int(next_color[0] * 255), int(next_color[1] * 255), int(next_color[2] * 255), alpha_value)
                    row_color[row_tag] = cell_color

                for col_position, col_value in enumerate(row_data):
                    item = QTableWidgetItem(str(col_value))
                    cell_color.setAlpha(alpha_value)
                    item.setBackground(cell_color)
                    self.csv_table.setItem(row_position, col_position, item)

        addRowsColored(row_list)
        addRowsColored(row_list_averaged, alpha_value=150)
        self.csv_data = row_list + row_list_averaged
        # Actualizar la vista de la tabla
        self.csv_table.resizeColumnsToContents()
        self.csv_table.resizeRowsToContents()
        
        # Sort by group
        self.csv_table.sortItems(self.csv_table.columnCount()-1, Qt.SortOrder.AscendingOrder)
        # log(f"[{self.__class__.__name__}] CSV data display completed")

    def sort_table(self, logical_index):
        # Manejar el evento de clic en el encabezado para ordenar la tabla
        self.csv_table.sortItems(logical_index, Qt.SortOrder.AscendingOrder)

