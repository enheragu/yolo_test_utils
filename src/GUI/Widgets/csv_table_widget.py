#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt widget to display data from trainig into a CSV table format
"""
import os
import itertools
from datetime import datetime
import csv
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QScrollArea, QSizePolicy, QVBoxLayout, QHBoxLayout, 
    QTableWidget, QTableWidgetItem, QDialog, QListWidget, QListWidgetItem,
    QPushButton, QCheckBox, QLabel, QAbstractItemView
)

from utils import log, bcolors
from utils.log_utils import logTable
from utils.log_utils import printDictKeys
from .check_box_widget import DatasetCheckBoxWidget, GroupCheckBoxWidget, BestGroupCheckBoxWidget
from GUI.gui_config import get_export_languages, get_language_suffix, get_language, translate_list, load_column_selection, save_column_selection, get_label_mappings


class ColumnSelectorDialog(QDialog):
    """
    Dialog to select and reorder columns for CSV export.
    Uses a list with checkboxes and drag-drop reordering.
    """
    def __init__(self, columns: list, selected_columns: Optional[list] = None, parent=None):
        """
        Args:
            columns: All available column names
            selected_columns: Currently selected columns in order. If None, all selected.
        """
        super().__init__(parent)
        self.setWindowTitle("Select Export Columns")
        self.setMinimumSize(400, 500)
        
        self.all_columns = columns
        if selected_columns is None:
            selected_columns = columns.copy()
        
        layout = QVBoxLayout(self)
        
        # Instructions
        layout.addWidget(QLabel("Check columns to export. Drag to reorder."))
        
        # List widget with checkboxes and drag-drop
        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        
        # Add items in order (selected first, then unselected)
        added = set()
        for col in selected_columns:
            if col in columns:
                self._add_column_item(col, checked=True)
                added.add(col)
        for col in columns:
            if col not in added:
                self._add_column_item(col, checked=False)
        
        layout.addWidget(self.list_widget)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        btn_layout.addWidget(deselect_all_btn)
        
        reset_btn = QPushButton("Reset Order")
        reset_btn.clicked.connect(self._reset_order)
        btn_layout.addWidget(reset_btn)
        
        layout.addLayout(btn_layout)
        
        # Apply/Cancel buttons
        action_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        action_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.accept)
        apply_btn.setDefault(True)
        action_layout.addWidget(apply_btn)
        
        layout.addLayout(action_layout)
    
    def _add_column_item(self, text: str, checked: bool = True):
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.list_widget.addItem(item)
    
    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Checked)
    
    def _deselect_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)
    
    def _reset_order(self):
        """Reset to original column order with all selected."""
        self.list_widget.clear()
        for col in self.all_columns:
            self._add_column_item(col, checked=True)
    
    def get_selected_columns(self) -> list:
        """Return list of checked columns in current order."""
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected


from Dataset.constants import dataset_options_keys

TABLE_DECIMAL_PRECISION="{:.5f}"

def getFusionType(dataset_type_str):
    len_sorted = sorted(dataset_options_keys, key=len, reverse=True)
    # log(f"[getFusionType] Trying to get fusion type from dataset type string '{dataset_type_str}' using keys: {len_sorted}")
    for key in len_sorted:
        if key in dataset_type_str:
            return key
    return "not-found"

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
        self.csv_data_averaged = [] # raw list with average and std data
        self.export_columns = load_column_selection()  # Load from session cache

    def configure_export_columns(self):
        if not self.csv_data:
            log(f"[{self.__class__.__name__}] No data to configure. Generate table first.", bcolors.WARNING)
            return

        all_columns = self.csv_data[0] if self.csv_data else []
        current_selection = self.export_columns if self.export_columns else all_columns

        dialog = ColumnSelectorDialog(all_columns, current_selection, self)
        if dialog.exec():
            self.export_columns = dialog.get_selected_columns()
            save_column_selection(self.export_columns)
            log(f"[{self.__class__.__name__}] Export columns configured: {len(self.export_columns)} columns selected")

    def _filter_columns(self, data: list) -> list:
        if not self.export_columns or not data:
            return data

        headers = data[0]
        col_indices = []
        for col in self.export_columns:
            if col in headers:
                col_indices.append(headers.index(col))

        if not col_indices:
            return data

        filtered = []
        for row in data:
            filtered_row = [row[i] if i < len(row) else '' for i in col_indices]
            filtered.append(filtered_row)

        return filtered

    def save_data(self, file_name=None):
        if not file_name or not self.csv_data:
            return

        filtered_data = self._filter_columns(self.csv_data)
        export_langs = get_export_languages()
        label_mappings = get_label_mappings()

        for lang in export_langs:
            lang_suffix = get_language_suffix(lang)
            output_file = f'{file_name}{lang_suffix}.csv'

            translated_data = [row.copy() for row in filtered_data]
            if translated_data:
                translated_data[0] = translate_list(filtered_data[0], lang)
                label_col = None
                try:
                    label_col = filtered_data[0].index('Label')
                except ValueError:
                    pass
                if label_col is not None:
                    for row in translated_data[1:]:
                        if label_col < len(row):
                            orig = row[label_col]
                            mapping = label_mappings.get(orig)
                            if isinstance(mapping, dict):
                                row[label_col] = mapping.get(lang, orig)

            with open(output_file, 'w', newline='') as file:
                log(f"[{self.__class__.__name__}] Summary CSV data stored in {output_file} ({len(translated_data[0])} columns)")
                writer = csv.writer(file)
                writer.writerows(translated_data)

            logTable(translated_data, os.path.dirname(file_name), f"{os.path.basename(file_name)}{lang_suffix}")

        if self.csv_data_averaged:
            output_path = os.path.dirname(file_name)
            filename = os.path.basename(file_name)
            with open(f'{file_name}_averaged.csv', 'w', newline='') as file:
                log(f"[{self.__class__.__name__}] Summary averaged CSV data stored in {file_name}_averaged.csv")
                writer = csv.writer(file)
                writer.writerows(self.csv_data_averaged)
            logTable(self.csv_data_averaged, output_path, f"{filename}_averaged")

    ## Overloadable function :)
    def getDataDictToPlot(self):
        data = {}
        
        for checkbox_list in self.dataset_checkboxes:
            if isinstance(checkbox_list, DatasetCheckBoxWidget):
                for key in checkbox_list.getChecked():           
                    data[key] = self.dataset_handler[key]
            elif isinstance(checkbox_list, GroupCheckBoxWidget):
                for group in checkbox_list.getChecked():
                    keys = [key for key in self.dataset_handler.keys() if key.startswith(f"{group}/")]
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

        row_list = [['Label', 'Model', 'Condition', 'Type', 'P', 'R', 'mAP50', 'mAP50-95', 'MR', 'LAMR', 'FPPI',
                     'Class', 'Dataset', 'Best epoch (index)', 'Train Duration (h)',
                     'Pretrained', 'Deterministic', 'Batch Size', 'Train Img', 'Val Img', 
                     'Instances', 'Num Classes', 'Dataset', 'Device', 'Date', 'Title', 'Group Key']]
        row_list_averaged = [[]] # Empty line, no title needed here

        info_dict = self.dataset_handler.getInfo()
        label_mappings = get_label_mappings()

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
                    
                    # Get plot label and apply mapping
                    plot_label = info_dict.get(key, {}).get('label', model)
                    mapping = label_mappings.get(plot_label)
                    if isinstance(mapping, dict):
                        plot_label = mapping.get(get_language(), plot_label)

                    # printDictKeys(data)
                    row_list.append([plot_label, model, condition, dataset_type[1],
                                    TABLE_DECIMAL_PRECISION.format(data_class.get('P', data_class.get(f"mP"))), 
                                    TABLE_DECIMAL_PRECISION.format(data_class.get('R', data_class.get(f"mR"))), 
                                    TABLE_DECIMAL_PRECISION.format(data_class['mAP50']), 
                                    TABLE_DECIMAL_PRECISION.format(data_class['mAP50-95']), 
                                    TABLE_DECIMAL_PRECISION.format(data['pr_data_best']['mr'][0]),
                                    TABLE_DECIMAL_PRECISION.format(data['pr_data_best']['lamr'][0]),
                                    TABLE_DECIMAL_PRECISION.format(data['pr_data_best']['fppi'][0]),
                                    class_type, 
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
            keys = [key for key in self.dataset_handler.keys() if key.startswith(f"{group}/")]
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
                    group_label = group.replace("train_based_variance_", "").replace(".yaml", "")
                    row_list_averaged.append([f"{group_label} ({tag})", model, condition, dataset_type[1],
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(p_vec_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(r_vec_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(mAP50_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(mAP50_95_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(mr_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(lamr_vec, axis = 0))}", 
                                f"({tag}) {TABLE_DECIMAL_PRECISION.format(function(fppi_vec, axis = 0))}", 
                                class_type, 
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
                                f"{(keys[0].split('/')[0] if keys else group)}/{tag}"])
                

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
        self.csv_data = row_list + [r for r in row_list_averaged if r]
        self.csv_data_averaged = row_list_averaged
        # Actualizar la vista de la tabla
        self.csv_table.resizeColumnsToContents()
        self.csv_table.resizeRowsToContents()
        
        # Sort by group
        self.csv_table.sortItems(self.csv_table.columnCount()-1, Qt.SortOrder.AscendingOrder)
        # log(f"[{self.__class__.__name__}] CSV data display completed")

    def sort_table(self, logical_index):
        # Manejar el evento de clic en el encabezado para ordenar la tabla
        self.csv_table.sortItems(logical_index, Qt.SortOrder.AscendingOrder)

