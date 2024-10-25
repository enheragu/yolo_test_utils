#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt tab view with all plot available to compare between different training runs
"""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QAction
from PyQt6.QtWidgets import QHBoxLayout, QWidget, QScrollArea, QSizePolicy, QVBoxLayout

from utils import parseYaml
from utils import log, bcolors
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
        self.layout.addWidget(self.options_widget,1)

        self.options_layout = QHBoxLayout()
        self.options_widget.setLayout(self.options_layout)
        self.options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Crear un widget que contendrá los grupos los botones
        self.buttons_widget = QWidget(self.options_widget)
        self.options_layout.addWidget(self.buttons_widget,1)
        self.buttons_layout = QVBoxLayout(self.buttons_widget)

        self.cursor = {}
        if tab_keys:
            self.tab_keys = tab_keys
            self.figure_tab_widget = PlotTabWidget(self.tab_keys)        
            self.layout.addWidget(self.figure_tab_widget,3)
        
        self.plot_background_img = True # Plots image with formula as background of each graph (if any)

    def toggle_options(self):
        # Cambiar el estado del check basado en si las opciones están visibles o no
        if self.options_widget.isVisible():
            self.options_widget.hide()
        else:
            self.options_widget.show()

    def toggle_checkbox_img_background(self, checked):
        # Convert the checkbox state to True or False
        self.plot_background_img = checked

    def update_view_and_menu(self, menu_list):
        archive_menu, view_menu, tools_menu, edit_menu = menu_list

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

        self.checkbox_action = QAction('Formula as background', self, checkable=True)
        self.checkbox_action.setChecked(True)  # Default plot_background_img state
        self.checkbox_action.triggered.connect(self.toggle_checkbox_img_background)

        # Add the checkbox action to the file menu
        view_menu.addAction(self.checkbox_action)
        
        self.edit_labels_action = QAction("Edit labels", self)
        self.edit_labels_action.triggered.connect(self.figure_tab_widget.edit_labels)
        edit_menu.addAction(self.edit_labels_action)

        self.edit_xlabels_action = QAction("Edit X labels", self)
        self.edit_xlabels_action.triggered.connect(self.figure_tab_widget.edit_xlabels)
        edit_menu.addAction(self.edit_xlabels_action)

        self.update_checkbox()

    def update_checkbox(self):
        raise NotImplementedError(f"This method has to be reimplemented in child {type(self).__name__} class")
    
    def save_plot(self):
        raise NotImplementedError(f"This method has to be reimplemented in child {type(self).__name__} class")

    def render_data(self):
        raise NotImplementedError(f"This method has to be reimplemented in child {type(self).__name__} class")
