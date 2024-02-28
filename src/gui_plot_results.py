#!/usr/bin/env python3
# encoding: utf-8

## NEEDS PYQT 5: pip install PyQt6  
## Using "export QT_DEBUG_PLUGINS=1" to see pluggin issues. Some might be missing. Install with apt
## Also needs to install lib: sudo apt install libxcb-cursor0
## might need an update -> pip install --upgrade pip

# If orphans are left behind just use:
# -> kill $(ps -aux | grep "gui_plot" | awk '{print $2}')

import sys
import argparse

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QMenu

from log_utils import log, bcolors
from GUI import DataSetHandler, TrainComparePlotter, TrainEvalPlotter, VarianceComparePlotter, CSVTablePlotter, SchedulerHandlerPlotter

sys.path.append('.')
import src # Imports __init__.py defined in paralel to this script

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
        
        train_eval_tab = VarianceComparePlotter(self.dataset_handler)
        self.tab_widget.addTab(train_eval_tab, f"Variance comparison")

        train_compare_tab = TrainComparePlotter(self.dataset_handler)
        self.tab_widget.addTab(train_compare_tab, f"Compare training data")

        train_eval_tab = TrainEvalPlotter(self.dataset_handler)
        self.tab_widget.addTab(train_eval_tab, f"Review training process")

        
        train_eval_tab = CSVTablePlotter(self.dataset_handler)
        self.tab_widget.addTab(train_eval_tab, f"Table")

        train_eval_tab = SchedulerHandlerPlotter()
        self.tab_widget.addTab(train_eval_tab, f"Scheduled tests")
                
        self.tab_widget.currentChanged.connect(self.update_view_menu)
        self.update_view_menu()  # Actualizar el menú "View" cuando se abre la ventana

    def reload_cached_datasetself(self):
        self.dataset_handler = DataSetHandler(False)
    
    def reload_raw_dataset(self):
        self.dataset_handler = DataSetHandler(True)

    def reload_incomplete_dataset(self):
        self.dataset_handler.reloadIncomplete()

    def update_view_menu(self):
        # Limpiar el menú "View"
        self.menuBar().clear()
        archive_menu = self.menuBar().addMenu('Archive')

        self.reload_datasets_menu = QMenu('Reload datasets')
        
        self.use_cached_datasets_action = QAction('Reload cached datasets')
        self.use_cached_datasets_action.triggered.connect( self.reload_cached_datasetself)
        self.reload_datasets_menu.addAction(self.use_cached_datasets_action)
        
        self.reload_from_raw_action = QAction('Reload all from raw')
        self.reload_from_raw_action.triggered.connect( self.reload_raw_dataset)
        self.reload_datasets_menu.addAction(self.reload_from_raw_action)

        self.reload_incomplete_action = QAction('Reload incomplete from raw')
        self.reload_incomplete_action.triggered.connect( self.reload_incomplete_dataset)
        self.reload_datasets_menu.addAction(self.reload_incomplete_action)

        archive_menu.addMenu(self.reload_datasets_menu)

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
    font = QApplication.font()
    font.setPointSize(10)  
    app.setFont(font)

    handleArguments()
    window = GUIPlotter()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    log(f"Start graph interface")
    main()