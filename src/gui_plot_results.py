#!/usr/bin/env python3
# encoding: utf-8

## NEEDS PYQT 5: pip install PyQt5  
## Using "export QT_DEBUG_PLUGINS=1" to see pluggin issues. Some might be missing. Install with apt
## Also needs to install lib: sudo apt install libxcb-cursor0
## might need an update -> pip install --upgrade pip

import os
import sys

from datetime import datetime

from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import csv

import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QCheckBox, QFileDialog, QGroupBox, QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import mplcursors

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
    return datasets

# Wrap function to be paralelized
def background_load_data(key):
    data = parseYaml(datasets[key]['path'])
    log(f"\t· Parsed {key} data")
    return data

class DataPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Training result analyzer")
        self.setGeometry(100, 100, 1200, 900)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout()

        # Check boxes grid configuration
        max_rows = 4 #int(len(datasets)/max_col + 0.5) # Round up
        max_col = 6
        row = 0

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.layout.addWidget(scroll_area, row, 0, 1, max_col - 2)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Hacerlo redimensionable solo horizontalmente

        # Crear un widget que contendrá los grupos de checkboxes
        scroll_widget = QWidget()
        scroll_widget.setMinimumHeight(max_rows) # por lo que sea poner un mínimo hace que evite el slider vertical lateral...
        scroll_area.setWidget(scroll_widget)
        scroll_layout = QGridLayout(scroll_widget)

        # Create group boxes to group checkboxes
        group_dict = {}
        last_group = ""
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

        # Crear un widget que contendrá los grupos los botones
        buttons_widget = QWidget(self.central_widget)
        self.layout.addWidget(buttons_widget, 0, 4, 1, 2)
        buttons_layout = QGridLayout(buttons_widget)

        ## Create a button to select all checkboxes from a condition
        self.select_all_day_button = QPushButton(" Select 'day' ", self)
        self.select_all_day_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.select_all_day_button.clicked.connect(lambda: self.select_all_checkboxes_cond('day'))
        self.select_all_night_button = QPushButton(" Select 'night' ", self)
        self.select_all_night_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.select_all_night_button.clicked.connect(lambda: self.select_all_checkboxes_cond('night'))
        self.select_all_all_button = QPushButton(" Select 'day-night' ", self)
        self.select_all_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.select_all_all_button.clicked.connect(lambda: self.select_all_checkboxes_cond('all'))

        ## Create a button to select all checkboxes
        self.select_all_button = QPushButton(" Select All ", self)
        self.select_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.select_all_button.clicked.connect(self.select_all_checkboxes)

        ## Create a button to deselect all checkboxes
        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.deselect_all_button.clicked.connect(self.deselect_all_checkboxes)

        ## Create a button to generate the plot
        self.plot_button = QPushButton(" Generate Plot ", self)
        self.plot_button.clicked.connect(self.plot_data)

        ## Create a button to save the plot
        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.clicked.connect(self.save_plot)

        buttons_layout.addWidget(self.select_all_day_button, 0, 0, 1, 1)
        buttons_layout.addWidget(self.select_all_night_button, 0, 1, 1, 1)
        buttons_layout.addWidget(self.select_all_all_button, 1, 0, 1, 1)
        buttons_layout.addWidget(self.select_all_button, 1, 1, 1, 1)
        buttons_layout.addWidget(self.deselect_all_button, 0, 2, 2, 1)
        buttons_layout.addWidget(self.plot_button, 2, 0, 1, 3)
        buttons_layout.addWidget(self.save_button, 3, 0, 1, 3)

        row += 1

        self.tab_canvas = {}
        self.ax = {}
        self.figure = {}
        self.cursor = {}
        self.tab_keys = ['PR Curve', 'P Curve', 'R Curve', 'F1 Curve']
        self.tabs = QTabWidget(self.central_widget)
        self.layout.addWidget(self.tabs, row, 0, len(self.tab_keys), max_col)

        # Crear pestañas para cada conjunto de datos
        for key in self.tab_keys:
            # Create a Matplotlib widget
            plt.rcParams.update({'font.size': 22})
            self.figure[key], self.ax[key] = plt.subplots()
            self.tab_canvas[key] = FigureCanvas(self.figure[key])
            
            # Agregar el lienzo a la pestaña
            tab = QWidget()
            tab.layout = QVBoxLayout()
            tab.layout.addWidget(self.tab_canvas[key] )
            tab.setLayout(tab.layout)
            self.tabs.addTab(tab, key)
        
        # Tab for CSV data
        self.csv_tab = QWidget()
        self.tabs.addTab(self.csv_tab, "Table")

        # Crear un QScrollArea dentro de la pestaña CSV
        csv_scroll_area = QScrollArea()
        csv_scroll_area.setWidgetResizable(True)
        self.csv_tab.layout = QVBoxLayout()
        self.csv_tab.layout.addWidget(csv_scroll_area)
        self.csv_tab.setLayout(self.csv_tab.layout)

        # Crear una tabla para mostrar datos CSV
        self.csv_table = QTableWidget()
        self.csv_table.setSortingEnabled(True)  # Habilitar la ordenación de columnas
        # self.csv_table.setDragDropMode(QTableWidget.InternalMove)  # Habilitar la reordenación de filas
        csv_scroll_area.setWidget(self.csv_table)

        self.csv_data = [] # raw list to be filled with the data which should be stored in csv file

        # Load data in background
        # Crear un ThreadPoolExecutor para cargar datos en segundo plano
        self.executor = ProcessPoolExecutor() # max_workers=12)
        self.futures = {key: self.executor.submit(background_load_data, key) for key in datasets}

        self.central_widget.setLayout(self.layout)


    def save_plot(self):
        # Open a file dialog to select the saving location
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)", options=options)

        if file_name:
            for key in self.figure.keys():
                # Save the plot to the selected location as a PNG image
                plot_name = file_name + "_" + key.replace(" ", "_")
                self.figure[key].savefig(plot_name, format='png')
                print(f"Plot saved to {plot_name}.png")
          
            with open(f'{file_name}.csv', 'w', newline='') as file:
                log(f"Summary CVS data in stored {file_name}.csv")
                writer = csv.writer(file)
                writer.writerows(self.csv_data)

    def plot_data(self):
        global parsed
        global datasets

        plot_data = {'PR Curve': {'py': 'py', 'xlabel': "Recall", "ylabel": 'Precision'},
                     'P Curve': {'py': 'p', 'xlabel': "Confidence", "ylabel": 'Precision'},
                     'R Curve': {'py': 'r', 'xlabel': "Confidence", "ylabel": 'Recall'},
                     'F1 Curve': {'py': 'f1', 'xlabel': "Confidence", "ylabel": 'F1'}}
        
        # Plotear los datos de los datasets seleccionados
        log(f"Parse YAML of selected datasets to plot, note that it can take some time:")
        for canvas_key in self.tab_keys:
            # Limpiar el gráfico anterior
            xlabel = plot_data[canvas_key]['xlabel']
            ylabel = plot_data[canvas_key]['ylabel']
            log(f"Plotting {ylabel}-{xlabel} Curve")

            self.ax[canvas_key].clear()
            for key in datasets:
                checkbox = datasets[key]['checkbox'] 
                if checkbox.isChecked():                
                    # Avoid continuous memory reading
                    if not key in parsed:
                        log(f"\t· Retrieve {key} data parsed from process executor")
                        data = self.futures[key].result() 
                        parsed[key] = data
                        # remove this retrieved key from dic
                        self.futures.pop(key)
                        if not self.futures:
                            # shutdown the process pool
                            self.executor.shutdown() # blocks
                    else:
                        # log(f"\t· Already parsed {key} data")
                        data = parsed[key]

                    py_tag = plot_data[canvas_key]['py']

                    try:
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

                    except KeyError as e:
                        log(f"Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)

            self.ax[canvas_key].set_xlabel(xlabel)
            self.ax[canvas_key].set_ylabel(ylabel)
            self.ax[canvas_key].set_xlim(0, 1)
            self.ax[canvas_key].set_ylim(0, 1)
            # self.ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
            self.ax[canvas_key].set_title(f'{ylabel}-{xlabel} Curve')
            
            # Configurar leyenda
            self.ax[canvas_key].legend()

            # Use a Cursor to interactively display the label for a selected line.
            self.cursor[canvas_key] = mplcursors.cursor(self.ax[canvas_key], hover=True)
            self.cursor[canvas_key].connect("add", lambda sel, xlabel=xlabel, ylabel=ylabel: sel.annotation.set(
                text=f"{sel.artist.get_label().split(' ')[0]}\n{xlabel}: {sel.target[0]:.2f}, {ylabel}: {sel.target[1]:.2f}",
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgrey', alpha=0.7)
            ))

            # Actualizar el gráfico
            self.tab_canvas[canvas_key].draw()

        log(f"Parsing and plot finished")

        self.load_table_data()
    
    def load_table_data(self):
        global parsed
        global datasets

        # Limpiar la tabla antes de cargar nuevos datos

        row_list = [['Model', 'Condition', 'Type', 'P', 'R', 'Train Images', 'Val Images', 'Instances', 'mAP50', 'mAP50-95', 'Class', 'Dataset', 'Best epoch (index)', 'Train Duration (h)', 'Pretrained', 'Date', 'Title']]

        for key in datasets:
            checkbox = datasets[key]['checkbox'] 
            if checkbox.isChecked():   
                try:
                    # Is done at the end of plotting, so data should already be in parsed dict             
                    data = parsed[key]

                    val_tag = f"validation_{data['val_epoch']-1}"

                    model = data[val_tag]['model'].split("/")[-1]
                    dataset = data[val_tag]['name'].split("/")[-1]
                    dataset_type = dataset.split("_")
                    
                    best_epoch = data['train_data']['epoch_best_fit_index']
                    train_duration = f"{data['train_data']['train_duration_h']:.2f}"
                    end_date_tag = data['train_data']['train_end_time']
                    train_images = data['n_images']['train']
                    val_images = data['n_images']['val']
                    pretrained = data['pretrained']

                    for class_type, data_class in data[val_tag]['data'].items():
                        if 'all' in class_type:
                            continue

                        # log(f"\t {data}")
                        date_tag = datetime.fromisoformat(end_date_tag).strftime('%Y-%m-%d_%H:%M:%S')
                        test_title = f'{model}_{dataset}_{date_tag}_{class_type}'
                        condition = 'night' if 'night' in dataset_type[0] else 'day'
                        row_list += [[model, condition, "_".join(dataset_type[1:]),
                                        "{:.4f}".format(data_class['P']), 
                                        "{:.4f}".format(data_class['R']), 
                                        train_images,
                                        val_images, 
                                        data_class['Instances'], 
                                        "{:.4f}".format(data_class['mAP50']), 
                                        "{:.4f}".format(data_class['mAP50-95']), 
                                        class_type, 
                                        dataset_type[0], 
                                        best_epoch,
                                        train_duration,
                                        pretrained,
                                        date_tag,
                                        test_title]]
                except KeyError as e:
                    log(f"Key error problem generating CSV for {key}. Row wont be generated. Missing key in data dict: {e}", bcolors.ERROR)

        self.csv_table.clear()
        self.csv_table.setRowCount(0)

        # Crear una nueva tabla
        self.csv_table.setColumnCount(len(row_list[0]))
        self.csv_table.setHorizontalHeaderLabels(row_list[0])
        
        # Conectar el evento sectionClicked del encabezado de la tabla
        self.csv_table.horizontalHeader().sectionClicked.connect(self.sort_table)

        # Agregar las filas a la tabla
        for row_position, row_data in enumerate(row_list[1:]):
            # row_position = self.csv_table.rowCount()
            self.csv_table.insertRow(row_position)
            for col_position, col_value in enumerate(row_data):
                item = QTableWidgetItem(str(col_value))
                self.csv_table.setItem(row_position, col_position, item)

        
        self.csv_data = row_list
        # Actualizar la vista de la tabla
        self.csv_table.resizeColumnsToContents()
        self.csv_table.resizeRowsToContents()
        
        log(f"CSV data display completed")

    def sort_table(self, logical_index):
        # Manejar el evento de clic en el encabezado para ordenar la tabla
        self.csv_table.sortItems(logical_index)

    def select_all_checkboxes(self):
        for key in datasets:
            checkbox = datasets[key]['checkbox'] 
            checkbox.setChecked(True)

    def select_all_checkboxes_cond(self, condition):
        for key in datasets:
            if condition in key:
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