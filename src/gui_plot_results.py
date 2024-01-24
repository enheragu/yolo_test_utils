#!/usr/bin/env python3
# encoding: utf-8

## NEEDS PYQT 5: pip install PyQt5  
## Using "export QT_DEBUG_PLUGINS=1" to see pluggin issues. Some might be missing. Install with apt
## Also needs to install lib: sudo apt install libxcb-cursor0
## might need an update -> pip install --upgrade pip

import sys
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QCheckBox, QFileDialog, QGroupBox, QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import os

from config_utils import yolo_output_path as test_path
from config_utils import log, bcolors, parseYaml
data_file_name = "results.yaml"
datasets = {}
parsed = {}

def find_results_file(search_path = test_path, file_name = data_file_name):
    log(f"Search all results.yaml files")
    global datasets

    for root, dirs, files in os.walk(search_path):
        if file_name in files:
            abs_path = os.path.join(root, file_name)
            if "validate" in abs_path: # Avoid validation tests, only training
                continue
            name = abs_path.split("/")[-3] + "/" + abs_path.split("/")[-2]
            datasets[name] = {'name': abs_path.split("/")[-2], 'path': abs_path, 'model': abs_path.split("/")[-3]}

    # # Order dataset by name
    myKeys = list(datasets.keys())
    myKeys.sort()
    datasets = {i: datasets[i] for i in myKeys}

class DataPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dataset Selector and Precision-Recall Plot")
        self.setGeometry(100, 100, 1200, 900)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout()

        # Check boxes grid configuration
        max_col = 6
        max_rows = int(len(datasets)/max_col + 0.5) # Round up
        row = 0

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Hacerlo redimensionable solo horizontalmente
        self.layout.addWidget(scroll_area, row, 0, 1, max_col)

        # Crear un widget que contendrá los grupos de checkboxes
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        scroll_layout = QGridLayout(scroll_widget)

        # Create group boxes to group checkboxes
        group_dict = {}
        last_group = ""
        grid = QGridLayout()
        iter = 0
        for dataset_name in datasets.keys():
            group_name = datasets[dataset_name]['model']
            test_name = datasets[dataset_name]['name']
            if group_name != last_group:
                iter = 0
                last_group = group_name
                group_dict[group_name] = QGroupBox(f"Model: {group_name}")
                group_dict[group_name].setLayout(QGridLayout())
                group_dict[group_name].setStyleSheet("font-weight: bold;")
                scroll_layout.addWidget(group_dict[group_name], 0, len(group_dict) - 1)

            checkbox = QCheckBox(test_name)
            checkbox.setStyleSheet("font-weight: normal;") # Undo the bold text from parent 
            datasets[dataset_name]['checkbox'] = checkbox
            row = iter % max_rows  # Divide en filas según max_rows
            col = iter // max_rows  # Divide en columnas según max_rows
            group_dict[group_name].layout().addWidget(checkbox, row, col)
            iter += 1

        row = row + 1

        ## Create a button to select all checkboxes
        self.select_all_button = QPushButton("Select All", self)
        self.layout.addWidget(self.select_all_button, row, 0, 1, int(max_col/2))
        self.select_all_button.clicked.connect(self.select_all_checkboxes)

        ## Create a button to deselect all checkboxes
        self.deselect_all_button = QPushButton("Deselect All", self)
        self.layout.addWidget(self.deselect_all_button, row, int(max_col/2), 1, int(max_col/2))
        self.deselect_all_button.clicked.connect(self.deselect_all_checkboxes)

        row += 1

        ## Create a button to generate the plot
        self.plot_button = QPushButton("Generate Plot", self)
        self.layout.addWidget(self.plot_button, row, 0, 1, max_col)
        self.plot_button.clicked.connect(self.plot_data)

        row += 1

        ## Create a button to save the plot
        self.save_button = QPushButton("Save Plot", self)
        self.layout.addWidget(self.save_button, row, 0, 1, max_col)
        self.save_button.clicked.connect(self.save_plot)

        row += 1

        self.tab_canvas = {}
        self.ax = {}
        self.figure = {}
        self.tab_keys = ['PR Curve', 'P Curve', 'R Curve', 'F1 Curve']
        self.tabs = QTabWidget(self.central_widget)
        self.layout.addWidget(self.tabs, row, 0, len(self.tab_keys), max_col)
        self.central_widget.setLayout(self.layout)

        # Crear pestañas para cada conjunto de datos
        for key in self.tab_keys:
            # Create a Matplotlib widget
            plt.rcParams.update({'font.size': 22})
            self.figure[key], self.ax[key] = plt.subplots()
            self.tab_canvas[key] = FigureCanvas(self.figure[key])
            # self.layout.addWidget(self.tab_canvas[key], row, 0, len(datasets), max_col) 
            # self.central_widget.setLayout(self.layout)
            
            # Agregar el lienzo a la pestaña
            tab = QWidget()
            tab.layout = QVBoxLayout()
            tab.layout.addWidget(self.tab_canvas[key] )
            tab.setLayout(tab.layout)
            self.tabs.addTab(tab, key)


    def save_plot(self):
        # Open a file dialog to select the saving location
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG Image", "", "PNG Images (*.png);;All Files (*)", options=options)

        if file_name:
            for key in self.figure.keys():
                # Save the plot to the selected location as a PNG image
                self.figure[key].savefig(file_name + key.replace(" ", "_"), format='png')
                print(f"Plot saved to {file_name}")

    def plot_data(self):
        global parsed

        plot_data = {'PR Curve': {'py': 'py', 'xlabel': "Recall", "ylabel": 'Precision'},
                     'P Curve': {'py': 'p', 'xlabel': "Confidence", "ylabel": 'Precision'},
                     'R Curve': {'py': 'r', 'xlabel': "Confidence", "ylabel": 'Recall'},
                     'F1 Curve': {'py': 'f1', 'xlabel': "Confidence", "ylabel": 'F1'}}
        
        # Plotear los datos de los datasets seleccionados
        log(f"Parse YAML of selected datasets to plot, note that it can take some time:")
        for canvas_key in self.tab_keys:
            # Limpiar el gráfico anterior
            self.ax[canvas_key].clear()
            for key in datasets:
                checkbox = datasets[key]['checkbox'] 
                if checkbox.isChecked():                
                    # Avoid continuous memory reading
                    if not key in parsed:
                        log(f"\t· Parse {key} data")
                        data = parseYaml(datasets[key]['path'])
                        parsed[key] = data
                    else:
                        log(f"\t· Already parsed {key} data")
                        data = parsed[key]

                    py_tag = plot_data[canvas_key]['py']
                    last_fit_tag = 'pr_data_' + str(data['pr_epoch'] - 1)
                    last_val_tag = 'validation_' + str(data['val_epoch'] - 1)

                    best_epoch = str(data['train_data']['epoch_best_fit_index'] + 1)

                    px = data[last_fit_tag]['px']
                    py = [data[last_fit_tag][py_tag]]
                    names = data[last_fit_tag]['names']
                    model = data[last_val_tag]['model'].split("/")[-1]
                    for py_list in py:
                        for i, y in enumerate(py_list):
                            self.ax[canvas_key].plot(px, y, linewidth=1, label=f"{datasets[key]['name']} ({model}) {names[i]} (best epoch: {best_epoch})")  # plot(confidence, metric)
            
            xlabel = plot_data[canvas_key]['xlabel']
            ylabel = plot_data[canvas_key]['ylabel']
            self.ax[canvas_key].set_xlabel(xlabel)
            self.ax[canvas_key].set_ylabel(ylabel)
            self.ax[canvas_key].set_xlim(0, 1)
            self.ax[canvas_key].set_ylim(0, 1)
            # self.ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
            self.ax[canvas_key].set_title(f'{ylabel}-{xlabel} Curve')
            
            # Configurar leyenda
            self.ax[canvas_key].legend()

            # Actualizar el gráfico
            self.tab_canvas[canvas_key].draw()

        log(f"Parsing and plot finished")


    def select_all_checkboxes(self):
        for key in datasets:
            checkbox = datasets[key]['checkbox'] 
            checkbox.setChecked(True)

    def deselect_all_checkboxes(self):
        for key in datasets:
            checkbox = datasets[key]['checkbox'] 
            checkbox.setChecked(False)

def main():
    app = QApplication(sys.argv)
    window = DataPlotter()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    find_results_file()
    log(f"Start graph interface")
    main()