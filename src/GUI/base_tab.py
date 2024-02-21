#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt tab view with all plot available to compare between different training runs
"""

import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QAction
from PyQt6.QtWidgets import QHBoxLayout, QWidget, QScrollArea, QSizePolicy, QVBoxLayout

from config_utils import parseYaml
from log_utils import log, bcolors
from .Widgets import PlotTabWidget


class BaseClassPlotter(QWidget):
    def __init__(self, dataset_handler, tab_keys):
        super().__init__()

        self.dataset_handler = dataset_handler

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # self.setWidgetResizable(True)  # Permitir que el widget se expanda
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.options_widget = QWidget(self)
        self.layout.addWidget(self.options_widget)

        self.options_layout = QHBoxLayout()
        self.options_widget.setLayout(self.options_layout)
        self.options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Crear un widget que contendrá los grupos los botones
        self.buttons_widget = QWidget(self.options_widget)
        self.options_layout.addWidget(self.buttons_widget,1)
        self.buttons_layout = QVBoxLayout(self.buttons_widget)

        self.cursor = {}
        if tab_keys:
            self.tab_keys = tab_keys
            self.figure_tab_widget = PlotTabWidget(self.tab_keys)        
            self.layout.addWidget(self.figure_tab_widget)
        
    
    def toggle_options(self):
        # Cambiar el estado del check basado en si las opciones están visibles o no
        if self.options_widget.isVisible():
            self.options_widget.hide()
        else:
            self.options_widget.show()

    def update_view_menu(self, archive_menu, view_menu, tools_menu):

        self.show_options_action = QAction('Show Options Tab', self, checkable=True)
        self.show_options_action.setShortcut(Qt.Key.Key_F11)
        self.show_options_action.setChecked(True) 
        self.show_options_action.triggered.connect(self.toggle_options)
        view_menu.addAction(self.show_options_action)

        self.plot_output_action = QAction("Plot Output", self)
        self.plot_output_action.setShortcut(Qt.Key.Key_F5)
        self.plot_output_action.triggered.connect(self.render_data)
        tools_menu.addAction(self.plot_output_action)

        self.save_output_action = QAction("Save Output", self)
        self.save_output_action.setShortcut(QKeySequence("Ctrl+S"))
        self.save_output_action.triggered.connect(self.save_plot)
        tools_menu.addAction(self.save_output_action)

    def save_plot(self):
        raise NotImplementedError("This method has to be reimplemented in child class")

    def render_data(self):
        raise NotImplementedError("This method has to be reimplemented in child class")
