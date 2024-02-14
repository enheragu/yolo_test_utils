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
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QPushButton, QFileDialog, QScrollArea, QSizePolicy, QVBoxLayout, QAction


import mplcursors

from config_utils import log, bcolors, parseYaml
from GUI.dataset_manager import DataSetHandler
from GUI.Widgets.check_box_widget import GroupCheckBoxWidget
from GUI.Widgets.csv_table_widget import TrainCSVDataTable


class CSVTablePlotter(QScrollArea):
    def __init__(self, dataset_handler, parent_window):
        super().__init__()

        self.parent_window = parent_window
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

        self.dataset_variance_checkboxes = GroupCheckBoxWidget(self.options_widget, dataset_handler, title = "Test groups:", max_rows = 3)
        self.options_layout.addWidget(self.dataset_variance_checkboxes,3)

        # Crear un widget que contendrá los grupos los botones
        buttons_widget = QWidget(self.options_widget)
        self.options_layout.addWidget(buttons_widget,1)
        buttons_layout = QVBoxLayout(buttons_widget)

        self.select_all_button = QPushButton(" Select All ", self)
        self.select_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.select_all_button.clicked.connect(self.dataset_variance_checkboxes.select_all)

        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.deselect_all_button.clicked.connect(lambda: (self.dataset_variance_checkboxes.deselect_all(), self.dataset_variance_checkboxes()))

        self.plot_button = QPushButton(" Generate Table ", self)
        self.plot_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_button.clicked.connect(self.render_data)

        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.save_button.clicked.connect(self.save_plot)

        # self.std_plot_widget = NumericSliderInputWidget('STD plot:')
        # self.deselect_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # buttons_layout.addWidget(self.std_plot_widget)

        buttons_layout.addWidget(self.select_all_button)
        buttons_layout.addWidget(self.deselect_all_button)
        buttons_layout.addWidget(self.plot_button)
        buttons_layout.addWidget(self.save_button)

        # Tab for CSV data
        self.csv_tab = TrainCSVDataTable(dataset_handler, self.dataset_variance_checkboxes)
        self.layout.addWidget(self.csv_tab)
   
    def toggle_options(self):
        # Cambiar el estado del check basado en si las opciones están visibles o no
        if self.options_widget.isVisible():
            self.options_widget.hide()
        else:
            self.options_widget.show()

    def update_view_menu(self):
        self.view_menu = self.parent_window.menuBar().addMenu('View')
        self.tools_menu = self.parent_window.menuBar().addMenu('Tools')


        self.show_options_action = QAction('Show Options Tab', self, checkable=True)
        self.show_options_action.setChecked(True) 
        self.show_options_action.triggered.connect(self.toggle_options)
        self.view_menu.addAction(self.show_options_action)

        # self.hide_options_action = QAction('Hide Options Tab', self)
        # self.hide_options_action.triggered.connect(self.options_widget.hide)
        # self.view_menu.addAction(self.hide_options_action)

        self.save_options_action = QAction("Save Output", self)
        self.save_options_action.triggered.connect(self.save_plot)
        self.tools_menu.addAction(self.save_options_action)
        
    def save_plot(self):
        # Open a file dialog to select the saving location
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)", options=options)

        if file_name:
            self.csv_tab.save_data(file_name)

    def render_data(self):
        self.csv_tab.load_table_data()