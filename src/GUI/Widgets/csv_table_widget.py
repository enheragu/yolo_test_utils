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
from GUI.dataset_manager import parse_dataset_name


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
        self.csv_data_averaged = [] # raw list with average and std data
        self.export_columns = load_column_selection()  # Load from session cache
        self.export_formats = None    # None = all (csv, txt, tex, html); set/list to filter
        self.export_languages = None  # None = use gui_config; list of lang codes to override

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

    # Rename map applied to the averaged table header after column filtering.
    # Keys are the internal column names (same as main table); values are the
    # display names shown in the exported averaged file.
    AVERAGED_HEADER_RENAME = {
        'Label':              'Method',
        'Type':               'Eq.V/T',
        'P':                  'P̄ ± σ',
        'R':                  'R̄ ± σ',
        'mAP50':              'mAP50̄ ± σ',
        'mAP50-95':           'mAP50-95̄ ± σ',
        'Best epoch (index)': 'Best epoch ± σ',
    }

    def _export_table(self, data, base_name, fmts, log_fmts, export_langs, label_mappings, rename_headers=None):
        """Filter, translate and write one table (csv + tex/txt/html) for each language.

        rename_headers: optional dict {old_col_name: new_col_name} applied after filtering.
        """
        filtered = self._filter_columns(data)
        if rename_headers and filtered:
            filtered = [filtered[0].copy()] + [row.copy() for row in filtered[1:]]
            filtered[0] = [rename_headers.get(h, h) for h in filtered[0]]
        output_path = os.path.dirname(base_name)
        base_filename = os.path.basename(base_name)
        for lang in export_langs:
            lang_suffix = get_language_suffix(lang)
            translated = [row.copy() for row in filtered]
            if translated:
                translated[0] = translate_list(filtered[0], lang)
                try:
                    label_col = filtered[0].index(rename_headers.get('Label', 'Label') if rename_headers else 'Label')
                except ValueError:
                    label_col = None
                if label_col is not None:
                    for row in translated[1:]:
                        if label_col < len(row):
                            orig = row[label_col]
                            mapping = label_mappings.get(orig)
                            if isinstance(mapping, dict):
                                row[label_col] = mapping.get(lang, orig)
            if 'csv' in fmts:
                out_csv = f'{base_name}{lang_suffix}.csv'
                with open(out_csv, 'w', newline='') as f:
                    log(f"[{self.__class__.__name__}] CSV stored in {out_csv} ({len(translated[0])} columns)")
                    csv.writer(f).writerows(translated)
            if log_fmts:
                logTable(translated, output_path, f"{base_filename}{lang_suffix}", formats=log_fmts)

    def save_data(self, file_name=None):
        if not file_name or not self.csv_data:
            return

        import re as _re
        fmts = set(self.export_formats) if self.export_formats is not None else {'csv', 'txt', 'tex', 'html'}
        log_fmts = fmts & {'txt', 'tex', 'html'} or None
        export_langs = self.export_languages if self.export_languages else get_export_languages()
        label_mappings = get_label_mappings()

        self._export_table(self.csv_data, file_name, fmts, log_fmts, export_langs, label_mappings)

        if len(self.csv_data_averaged) > 1:  # more than just the header row
            # Strip mode suffix so averaged tables are mode-independent
            avg_base = _re.sub(r'_(best|p\d+)$', '', file_name)
            avg_base = f'{avg_base}_averaged'
            self._export_table(self.csv_data_averaged, avg_base, fmts, log_fmts, export_langs, label_mappings,
                               rename_headers=self.AVERAGED_HEADER_RENAME)

    ## Overloadable function :)
    def getDataDictToPlot(self):
        data = {}
        
        for checkbox_list in self.dataset_checkboxes:
            if isinstance(checkbox_list, DatasetCheckBoxWidget):
                for key in checkbox_list.getChecked():
                    data[key] = self.dataset_handler[key]
            elif isinstance(checkbox_list, BestGroupCheckBoxWidget):
                # getChecked() returns leaf trial keys (best per group), not group prefixes
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
                data += list(checkbox_list.getCheckedGroups().keys())
            elif isinstance(checkbox_list, GroupCheckBoxWidget):
                data += checkbox_list.getChecked()
            else:
                log(f"[{self.__class__.__name__}] CheckBox object instanced not recogniced.", bcolors.ERROR)
        return data

    def load_table_data(self):
        # Limpiar la tabla antes de cargar nuevos datos

        row_list = [['Label', 'Model', 'Condition', 'Type', 'P', 'R', 'mAP50', 'mAP50-95', 'MR', 'LAMR', 'FPPI',
                     'Class', 'Best epoch (index)', 'Train Duration (h)',
                     'Pretrained', 'Deterministic', 'Batch Size', 'Train Img', 'Val Img',
                     'Instances', 'Num Classes', 'Dataset', 'Device', 'Date', 'Title', 'Group Key']]
        row_list_averaged = [row_list[0]]  # Same header so _filter_columns works with preset column names

        info_dict = self.dataset_handler.getInfo()
        label_mappings = get_label_mappings()

        for key, data in self.getDataDictToPlot().items():
            if not data:
                continue
            try:
                model = data['validation_best']['model'].split("/")[-1]
                dataset = data['validation_best']['name'].split("/")[-1]
                _ds_name, method_type, parsed_condition = parse_dataset_name(dataset)

                end_date_tag = data['train_data']['train_end_time']
                for class_type, data_class in data['validation_best']['data'].items():
                    if 'all' in class_type:
                        continue

                    date_tag = datetime.fromisoformat(end_date_tag).strftime('%Y-%m-%d_%H:%M:%S')
                    test_title = f'{model}_{dataset}_{date_tag}_{class_type}'
                    # Prefer parsed condition; fall back to legacy substring match in first token
                    if parsed_condition:
                        condition = parsed_condition
                    else:
                        first = dataset.split('_')[0]
                        if 'night' in first:   condition = 'night'
                        elif 'day' in first:   condition = 'day'
                        elif 'all' in first:   condition = 'all'
                        else:                  condition = "Unknown"
                    
                    # Get plot label and apply mapping
                    plot_label = info_dict.get(key, {}).get('label', model)
                    mapping = label_mappings.get(plot_label)
                    if isinstance(mapping, dict):
                        plot_label = mapping.get(get_language(), plot_label)
                    if '/' in key:
                        group = key.rsplit('/', 1)[0]
                        n = sum(1 for k in self.dataset_handler.keys() if k.startswith(f"{group}/"))
                        if n > 1:
                            plot_label = f"{plot_label} (N={n})"

                    row_list.append([plot_label, model, condition, method_type,
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
            _ds_name, method_type, parsed_condition = parse_dataset_name(dataset)

            for class_type, data_class in data['validation_best']['data'].items():
                if 'all' in class_type:
                    continue

                date_tag = "-"
                test_title = f'{model}_{dataset}_{class_type}'
                if parsed_condition:
                    condition = parsed_condition
                else:
                    first = dataset.split('_')[0]
                    if 'night' in first:   condition = 'night'
                    elif 'day' in first:   condition = 'day'
                    elif 'all' in first:   condition = 'all'
                    else:                  condition = "Unknown"

                def _pm(vec):
                    return f"{np.mean(vec):.5f} ± {np.std(vec):.5f}"
                def _pm2(vec):
                    return f"{np.mean(vec):.2f} ± {np.std(vec):.2f}"

                eq_map = {'rgb_equalization': 'RGB', 'th_equalization': 'TH', 'no_equalization': '-'}
                eq_label = eq_map.get(group.split('/')[0], group.split('/')[0])

                plot_label = info_dict.get(keys[0] if keys else group, {}).get('label', group.split('/')[-1])
                mapping = label_mappings.get(plot_label)
                if isinstance(mapping, dict):
                    plot_label = mapping.get(get_language(), plot_label)
                plot_label = f"{plot_label} (N={len(keys)})"

                row_list_averaged.append([plot_label, model, condition, eq_label,
                            _pm(p_vec_vec),
                            _pm(r_vec_vec),
                            _pm(mAP50_vec),
                            _pm(mAP50_95_vec),
                            _pm(mr_vec),
                            _pm(lamr_vec),
                            _pm(fppi_vec),
                            class_type,
                            _pm2(bestfit_epoch_vec),
                            _pm2(train_duration_vec),
                            data['pretrained'],
                            data['deterministic'],
                            data['batch'],
                            int(data['n_images']['train']),
                            int(data['n_images']['val']),
                            _pm2(instances_vec),
                            data['n_classes'],
                            data['dataset_tag'],
                            data['device_type'],
                            date_tag,
                            test_title,
                            keys[0].split('/')[0] if keys else group])
                

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
        self.csv_data = row_list
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

