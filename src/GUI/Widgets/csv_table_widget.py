#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt widget to display data from trainig into a CSV table format
"""

import itertools
from datetime import datetime
import csv

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QScrollArea, QSizePolicy, QVBoxLayout, QTableWidget, QTableWidgetItem

from log_utils import log, bcolors
from .check_box_widget import DatasetCheckBoxWidget, GroupCheckBoxWidget

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
            self.figure_tab_widget.saveFigures(file_name)
          
            with open(f'{file_name}.csv', 'w', newline='') as file:
                log(f"[{self.__class__.__name__}] Summary CVS data in stored {file_name}.csv")
                writer = csv.writer(file)
                writer.writerows(self.csv_data)

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

    def load_table_data(self):
        # Limpiar la tabla antes de cargar nuevos datos

        row_list = [['Model', 'Condition', 'Type', 'P', 'R', 'mAP50', 'mAP50-95', 'Class', 'Dataset', 'Best epoch (index)', 'Train Duration (h)', 'Pretrained', 'Deterministic', 'Batch Size', 'Train Img', 'Val Img', 'Instances', 'Num Classes', 'Dataset', 'Device', 'Date', 'Title', 'Group Key']]

        for key, data in self.getDataDictToPlot().items():
            try:
                model = data['validation_best']['model'].split("/")[-1]
                dataset = data['validation_best']['name'].split("/")[-1]
                dataset_type = dataset.split("_")
                
                best_epoch = data['train_data']['epoch_best_fit_index']
                train_duration = f"{data['train_data']['train_duration_h']:.2f}"
                end_date_tag = data['train_data']['train_end_time']
                train_images = int(data['n_images']['train'])
                val_images = int(data['n_images']['val'])
                pretrained = data['pretrained']
                deterministic = data['deterministic']

                for class_type, data_class in data['validation_best']['data'].items():
                    if 'all' in class_type:
                        continue

                    # log(f"\t {data}")
                    date_tag = datetime.fromisoformat(end_date_tag).strftime('%Y-%m-%d_%H:%M:%S')
                    test_title = f'{model}_{dataset}_{date_tag}_{class_type}'
                    condition = 'night' if 'night' in dataset_type[0] else 'day'
                    row_list += [[model, condition, "_".join(dataset_type[1:]),
                                    "{:.4f}".format(data_class['P']), 
                                    "{:.4f}".format(data_class['R']), 
                                    "{:.4f}".format(data_class['mAP50']), 
                                    "{:.4f}".format(data_class['mAP50-95']), 
                                    class_type, 
                                    dataset_type[0], 
                                    best_epoch,
                                    train_duration,
                                    pretrained,
                                    deterministic,
                                    data['batch'],
                                    train_images,
                                    val_images, 
                                    data_class['Instances'], 
                                    data['n_classes'],
                                    data['dataset_tag'],
                                    data['device_type'],
                                    date_tag,
                                    test_title,
                                    key]]
            except KeyError as e:
                log(f"[{self.__class__.__name__}] Key error problem generating CSV for {key}. Row wont be generated. Missing key in data dict: {e}", bcolors.ERROR)

        self.csv_table.clear()
        self.csv_table.setRowCount(0)

        # Crear una nueva tabla
        self.csv_table.setColumnCount(len(row_list[0]))
        self.csv_table.setHorizontalHeaderLabels(row_list[0])
        
        # Conectar el evento sectionClicked del encabezado de la tabla
        self.csv_table.horizontalHeader().sectionClicked.connect(self.sort_table)


        colors_list = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
        color_iterator = itertools.cycle(colors_list)
        row_color = {}
        # Agregar las filas a la tabla
        for row_position, row_data in enumerate(row_list[1:]):
            # row_position = self.csv_table.rowCount()
            self.csv_table.insertRow(row_position)
            
            ## Add colors
            ## row_tag Filters by gout/condition_type
            row_tag = row_data[-1].split("/")[0]+"/"+row_data[1]
            if row_tag in row_color:
                cell_color = row_color[row_tag]
            else:
                next_color = next(color_iterator)
                if isinstance(next_color, str):
                    next_color = to_rgba(next_color)
                alpha_value = 20  # transparency range between 0 and 255
                cell_color = QColor(int(next_color[0] * 255), int(next_color[1] * 255), int(next_color[2] * 255), alpha_value)
                row_color[row_tag] = cell_color

            for col_position, col_value in enumerate(row_data):
                item = QTableWidgetItem(str(col_value))
                item.setBackground(cell_color)
                self.csv_table.setItem(row_position, col_position, item)

        
        self.csv_data = row_list
        # Actualizar la vista de la tabla
        self.csv_table.resizeColumnsToContents()
        self.csv_table.resizeRowsToContents()
        
        log(f"[{self.__class__.__name__}] CSV data display completed")

    def sort_table(self, logical_index):
        # Manejar el evento de clic en el encabezado para ordenar la tabla
        self.csv_table.sortItems(logical_index)

