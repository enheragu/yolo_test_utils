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


import csv
import math

import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QCheckBox, QFileDialog, QGroupBox, QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QTableWidget, QTableWidgetItem


import mplcursors

from config_utils import log, bcolors, parseYaml
from GUI.dataset_manager import DataSetHandler
from GUI.train_compare_tab import TrainComparePlotter
from GUI.train_eval_tab import TrainEvalPlotter
from GUI.variance_compare_tab import VarianceComparePlotter
from GUI.csv_table_tab import CSVTablePlotter

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
        
        self.dataset_handler = DataSetHandler()
        train_compare_tab = TrainComparePlotter(self.dataset_handler)
        self.tab_widget.addTab(train_compare_tab, f"Compare training data")

        train_eval_tab = TrainEvalPlotter(self.dataset_handler)
        self.tab_widget.addTab(train_eval_tab, f"Review training process")

        train_eval_tab = VarianceComparePlotter(self.dataset_handler)
        self.tab_widget.addTab(train_eval_tab, f"Variance comparison")
        
        train_eval_tab = CSVTablePlotter(self.dataset_handler)
        self.tab_widget.addTab(train_eval_tab, f"Table")
        

def main():
    app = QApplication(sys.argv)
    window = GUIPlotter()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    log(f"Start graph interface")
    main()