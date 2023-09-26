#!/usr/bin/env python3
# encoding: utf-8

## NEEDS PYQT 5: pip install PyQt5  
## Using "export QT_DEBUG_PLUGINS=1" to see pluggin issues. Some might be missing. Install with apt
## Also needs to install lib: sudo apt install libxcb-cursor0
## might need an update -> pip install --upgrade pip

import sys
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QCheckBox, QFileDialog
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
            name = abs_path.split("/")[-2]
            datasets[name] = abs_path

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

        # Create checkboxes in a grid
        self.checkbox_matrix = []
        max_col = 4
        max_rows = int(len(datasets)/max_col + 0.5) # Round up
        col = 0
        row = 0
        for iter, dataset_name in enumerate(datasets.keys()):
            if row >= max_rows:
                col += 1
                row = 0

            checkbox = QCheckBox(dataset_name)
            self.checkbox_matrix.append(checkbox)
            self.layout.addWidget(checkbox, row, col)
            row += 1

        row = max_rows + 1

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

        # Create a Matplotlib widget
        plt.rcParams.update({'font.size': 22})
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas, row, 0, len(datasets), max_col) 
        self.central_widget.setLayout(self.layout)

    def save_plot(self):
        # Open a file dialog to select the saving location
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG Image", "", "PNG Images (*.png);;All Files (*)", options=options)

        if file_name:
            # Save the plot to the selected location as a PNG image
            self.figure.savefig(file_name, format='png')
            print(f"Plot saved to {file_name}")

    def plot_data(self):
        global parsed
        # Limpiar el gráfico anterior
        self.ax.clear()

        # Plotear los datos de los datasets seleccionados
        log(f"Parse YAML of selected datasets to plot, note that it can take some time:")
        for checkbox in self.checkbox_matrix:
            if checkbox.isChecked():
                dataset_name = checkbox.text()
                
                # Avoid continuous memory reading
                if not dataset_name in parsed:
                    log(f"\t· Parse {dataset_name} data")
                    data = parseYaml(datasets[dataset_name])
                    parsed[dataset_name] = data
                else:
                    log(f"\t· Already parsed {dataset_name} data")
                    data = parsed[dataset_name]

                px = data['pr_data']['px']
                py = [data['pr_data']['py']]
                ap = [data['pr_data']['ap']]
                # f1 = [test['pr_data']['f1'] for test in data],
                # p = [test['pr_data']['p'] for test in data],
                # r = [test['pr_data']['r'] for test in data],
                names = data['pr_data']['names']
                labels = [data['test'].split("/")[-1].split("_")[-1]],
                # path = path,
                # title_name = f"{key}  -  "
                ap_labels = ap
                if ap_labels == []:
                    ap_labels = [""]*len(labels)
                
                for py_list in py:
                    for i, y in enumerate(py_list):
                        self.ax.plot(px, y, linewidth=1, label=f'{dataset_name} {names[i]}')  # plot(confidence, metric)
        
        self.ax.set_xlabel("Recall")
        self.ax.set_ylabel("Precision")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        # self.ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        self.ax.set_title(f'Precision-Recall Curve')
        
        # Configurar leyenda
        self.ax.legend()

        log(f"Parsing and plot finished")

        # Actualizar el gráfico
        self.canvas.draw()

    def select_all_checkboxes(self):
        for checkbox in self.checkbox_matrix:
            checkbox.setChecked(True)

    def deselect_all_checkboxes(self):
        for checkbox in self.checkbox_matrix:
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