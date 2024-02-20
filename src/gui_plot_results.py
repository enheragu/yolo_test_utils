#!/usr/bin/env python3
# encoding: utf-8

## NEEDS PYQT 5: pip install PyQt5  
## Using "export QT_DEBUG_PLUGINS=1" to see pluggin issues. Some might be missing. Install with apt
## Also needs to install lib: sudo apt install libxcb-cursor0
## might need an update -> pip install --upgrade pip

# If orphans are left behind just use:
# -> kill $(ps -aux | grep "gui_plot" | awk '{print $2}')
import os
import sys

from datetime import datetime

import argparse
import csv
import math

import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction, QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QCheckBox, QFileDialog, QGroupBox, QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QTableWidget, QTableWidgetItem

import mplcursors

sys.path.append('.')
import src # Imports __init__.py defined in paralel to this script


from config_utils import parseYaml
from log_utils import log, bcolors
from GUI.dataset_manager import DataSetHandler
from GUI.train_compare_tab import TrainComparePlotter
from GUI.train_eval_tab import TrainEvalPlotter
from GUI.variance_compare_tab import VarianceComparePlotter
from GUI.csv_table_tab import CSVTablePlotter

## Data YAML cached for faster data loading
# Update means that previous will be overwriten
update_cache = False

class GUIPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training result analyzer")
        self.setGeometry(100, 100, 1400, 1000)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        self.dataset_handler = DataSetHandler(update_cache)
        train_compare_tab = TrainComparePlotter(self.dataset_handler)
        self.tab_widget.addTab(train_compare_tab, f"Compare training data")

        train_eval_tab = TrainEvalPlotter(self.dataset_handler)
        self.tab_widget.addTab(train_eval_tab, f"Review training process")

        train_eval_tab = VarianceComparePlotter(self.dataset_handler)
        self.tab_widget.addTab(train_eval_tab, f"Variance comparison")
        
        train_eval_tab = CSVTablePlotter(self.dataset_handler)
        self.tab_widget.addTab(train_eval_tab, f"Table")
        
        
        self.tab_widget.currentChanged.connect(self.update_view_menu)
        self.update_view_menu()  # Actualizar el menú "View" cuando se abre la ventana

    def update_dataset_handler(self):
        self.dataset_handler = DataSetHandler(update_cache)
        self.cached_option_action.setChecked(False) 
    
    def toggle_cached(self):
        global update_cache
        update_cache = not update_cache

    def update_view_menu(self):
        # Limpiar el menú "View"
        self.menuBar().clear()
        archive_menu = self.menuBar().addMenu('Archive')

        self.reload_datasets_action = QAction('Reload datasets')
        self.reload_datasets_action.triggered.connect(self.update_dataset_handler)
        archive_menu.addAction(self.reload_datasets_action)

        self.cached_option_action = QAction('Use cached datasets', self, checkable=True)
        self.cached_option_action.setChecked(True) 
        self.cached_option_action.triggered.connect(self.toggle_cached)
        archive_menu.addAction(self.cached_option_action)
        
        view_menu = self.menuBar().addMenu('View')
        tools_menu = self.menuBar().addMenu('Tools')

        # Obtener la pestaña/tab actual
        current_tab_widget = self.tab_widget.currentWidget()
        current_tab_widget.update_view_menu(archive_menu, view_menu, tools_menu)

def handleArguments():
    global update_cache
    parser = argparse.ArgumentParser(description="GUI to review training results.")
    parser.add_argument('--update_cache', action='store_true', help='Actualizar archivos de caché si es verdadero')

    # Parsear los argumentos de la línea de comandos
    args = parser.parse_args()

    update_cache = args.update_cache
    log(f"Update cache flag is: {update_cache}")

def main():
    app = QApplication(sys.argv)
    handleArguments()
    window = GUIPlotter()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    log(f"Start graph interface")
    main()