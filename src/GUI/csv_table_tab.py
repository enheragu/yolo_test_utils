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
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QFileDialog, QSizePolicy

from utils import log, bcolors
from GUI.base_tab import BaseClassPlotter
from GUI.base_tab import BaseClassPlotter
from GUI.dataset_manager import DataSetHandler
from GUI.Widgets.check_box_widget import GroupCheckBoxWidget
from GUI.Widgets.csv_table_widget import TrainCSVDataTable

tab_keys = []

class CSVTablePlotter(BaseClassPlotter):
    TAB_ID = 'csv_table'

    def __init__(self, dataset_handler):
        super().__init__(dataset_handler, tab_keys)

        self.dataset_variance_checkboxes = GroupCheckBoxWidget(self.options_widget, dataset_handler, title = "Test groups:", max_rows = 3)
        self.options_layout.insertWidget(0, self.dataset_variance_checkboxes,3)

        self.select_all_button = QPushButton(" Select All ", self)
        self.select_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.select_all_button.clicked.connect(self.dataset_variance_checkboxes.select_all)

        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.deselect_all_button.clicked.connect(self.dataset_variance_checkboxes.deselect_all)

        self.plot_button = QPushButton(" Generate Table ", self)
        self.plot_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_button.clicked.connect(self.render_data)

        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.save_button.clicked.connect(self.save_plot)

        self.buttons_layout.addWidget(self.select_all_button)
        self.buttons_layout.addWidget(self.deselect_all_button)
        self.buttons_layout.addWidget(self.plot_button)
        self.buttons_layout.addWidget(self.save_button)

        # Tab for CSV data
        self.csv_tab = TrainCSVDataTable(dataset_handler, self.dataset_variance_checkboxes)
        self.layout.addWidget(self.csv_tab, 3)
    
        log(f"[{self.__class__.__name__}] Initialized.")

    def update_checkbox(self):
        self.dataset_variance_checkboxes.update_checkboxes()

    def save_plot(self):
        # Open a file dialog to select the saving location
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)")

        if file_name:
            self.csv_tab.save_data(file_name)

    def render_data(self):
        self.csv_tab.load_table_data()
        log(f"[{self.__class__.__name__}] Table updated")

    # ── Preset hooks ──────────────────────────────────────────────────
    # This tab dumps the full table (every trial as its own row, plus a
    # mean ± std averaged file). It has a single group selector and no
    # mode/class concept, so those preset fields are left empty.
    def _collect_tab_state(self):
        return {
            'mode':                None,
            'class_key':           'all',
            'dataset_keys':        [],
            'variance_group_keys': self.dataset_variance_checkboxes.getChecked(),
        }

    def _apply_tab_state(self, preset):
        # The single selector holds every test group. A '*' / 'all' value selects
        # them all (the "Select All" button); explicit keys may be listed under
        # 'variance_groups' or 'datasets' (merged here, since there is one widget).
        sel = preset.get('selection', {}) or {}

        def _is_wildcard(v):
            return isinstance(v, str) and v.strip().lower() in ('*', 'all')

        ds = sel.get('datasets', [])
        vg = sel.get('variance_groups', [])
        if _is_wildcard(ds) or _is_wildcard(vg):
            self.dataset_variance_checkboxes.select_all()
        else:
            self.dataset_variance_checkboxes.setCheckedKeys(list(ds) + list(vg))