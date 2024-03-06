#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt tab view with all plot available to compare between different training runs
"""

import itertools
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm
from PyQt6.QtWidgets import QPushButton, QFileDialog, QSizePolicy

import mplcursors

from utils import log, bcolors
from GUI.base_tab import BaseClassPlotter
from GUI.Widgets import DatasetCheckBoxWidget, GroupCheckBoxWidget, TrainCSVDataTable

tab_keys = ['PR Curve', 'P Curve', 'R Curve', 'F1 Curve', 'mAP50', 'mAP50-95', 'P', 'R']

class VarianceComparePlotter(BaseClassPlotter):
    def __init__(self, dataset_handler):
        super().__init__(dataset_handler, tab_keys)

        self.dataset_variance_checkboxes = GroupCheckBoxWidget(self.options_widget, dataset_handler, include = "variance_", title = f"Variance analysis sets:", title_filter=["variance_"])
        self.options_layout.insertWidget(0, self.dataset_variance_checkboxes,3)

        self.dataset_train_checkboxes = DatasetCheckBoxWidget(self.options_widget, dataset_handler, title_filter=["train_based_"])
        self.options_layout.insertWidget(0, self.dataset_train_checkboxes,3)
        
        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.deselect_all_button.clicked.connect(lambda: (self.dataset_train_checkboxes.deselect_all(), self.dataset_variance_checkboxes.deselect_all()))

        self.plot_button = QPushButton(" Generate Plot ", self)
        self.plot_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_button.clicked.connect(self.render_data)

        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.save_button.clicked.connect(self.save_plot)

        self.buttons_layout.addWidget(self.deselect_all_button)
        self.buttons_layout.addWidget(self.plot_button)
        self.buttons_layout.addWidget(self.save_button)

        # Tab for CSV data
        self.csv_tab = TrainCSVDataTable(dataset_handler, [self.dataset_train_checkboxes,self.dataset_variance_checkboxes])
        self.figure_tab_widget.addTab(self.csv_tab, "Table")
    
    def update_checkbox(self):
        self.dataset_train_checkboxes.update_checkboxes()
        self.dataset_variance_checkboxes.update_checkboxes()

    def save_plot(self):
        # Open a file dialog to select the saving location
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)")

        if file_name:
            self.figure_tab_widget.saveFigures(file_name)
            self.csv_tab.save_data(file_name)

    def render_data(self):
        self.plot_loss_metrics_data()
        self.csv_tab.load_table_data()
        log(f"[{self.__class__.__name__}] Plot and table updated")

    def plot_loss_metrics_data(self):

        # PY is an interpolated versino to plot it with a consistent px value. That's why
        # it is used here instead of plotting raw p-r values averaged
        plot_data = {'PR Curve': {'py': 'py', 'xlabel': "Recall", "ylabel": 'Precision'},
                     'P Curve': {'py': 'p', 'xlabel': "Confidence", "ylabel": 'Precision'},
                     'R Curve': {'py': 'r', 'xlabel': "Confidence", "ylabel": 'Recall'},
                     'F1 Curve': {'py': 'f1', 'xlabel': "Confidence", "ylabel": 'F1'}}
        
        # Borrar gráficos previos
        self.figure_tab_widget.clear()

        # Plotear los datos de los datasets seleccionados
        # log(f"Parse YAML of selected datasets to plot, note that it can take some time:")
        for canvas_key in self.tab_keys:
            
            if canvas_key == 'mAP50' or canvas_key == 'mAP50-95' or canvas_key == 'P' or canvas_key == 'R':
                colors_list = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
                color_iterator = itertools.cycle(colors_list)
                # Limpiar el gráfico anterior
                # xlabel = canvas_key
                # ylabel = "Probability"
                
                bin_size = 9
                ax = self.figure_tab_widget[canvas_key].add_axes([0.08, 0.08, 0.84, 0.86])
                for index, group in enumerate(self.dataset_variance_checkboxes.getChecked()):
                    keys = [key for key in self.dataset_handler.keys() if group in key]
                    data_y = np.array([])
                    for key in keys:
                        data = self.dataset_handler[key]
                        if not data:
                            continue
                        try:
                            data_y = np.append(data_y, data['validation_best']['data']['all'][canvas_key])
                        except KeyError as e:
                            log(f"[{self.__class__.__name__}] Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                            self.dataset_handler.markAsIncomplete(key)
                            continue
                    
                    if np.size(data_y) == 0:
                        log(f"[{self.__class__.__name__}] Empty data_y vector, nothing to plot for {group}", bcolors.ERROR)
                        continue
                    
                    
                    # If all data are exactly the same plotting will have issues (STD=0)
                    # Add little noise for plotting
                    if np.all(data_y == data_y[0]):
                        data_y += np.random.normal(scale=0.00001, size=len(data_y))

                    # log(f"{group}: {len(data_y) = }")
                    next_color = next(color_iterator)
                    label = group.replace("train_based_variance_", "").replace(".yaml", "")    
                    ax.hist(data_y, bins=bin_size, density=True, alpha=0.5, label=f'{label}; n = {len(data_y)}', color=next_color, edgecolor=next_color)
                    
                    mean = np.mean(data_y)
                    std = np.std(data_y)

                    if std >0.001:       
                        
                        x = np.linspace(mean-std*4, mean+std*4, 100)
                        y = norm.pdf(x, mean, std)
                        ax.plot(x, y, linestyle='--', linewidth=2, color=next_color)

                        for pos in np.arange(mean-std*2, mean+std*2, std):
                            ax.axvline(x=pos, color=next_color, linewidth=1)
                    else:
                        log(f"[{self.__class__.__name__}] STD of data is <0 ({std = }) for {canvas_key} plot with n={len(data_y)} for {group}", bcolors.ERROR)

                ax.annotate(f'Note: Each set of {canvas_key} data is discretized into {bin_size} bins.',
                                xy = (0.995, 0.015), xycoords='axes fraction',
                                ha='right', va="center", fontsize=8)
                # ax.set_yscale('log')
                # ax.set_ylabel('log scale')
                ax.set_title(f'{canvas_key} distribution')
            else:
                if canvas_key not in plot_data:
                    continue

                # Limpiar el gráfico anterior
                xlabel = plot_data[canvas_key]['xlabel']
                ylabel = plot_data[canvas_key]['ylabel']
                # log(f"Plotting {ylabel}-{x, 'mAP50', 'mAP50-95'label} Curve")

                # ax = self.figure_tab_widget[canvas_key].add_subplot(111) #add_axes([0.08, 0.08, 0.84, 0.86])
                ax = self.figure_tab_widget[canvas_key].add_axes([0.08, 0.08, 0.84, 0.86])
                py_tag = plot_data[canvas_key]['py']

                def getLastEpochData(key_data, raw_data_dict):
                    px = raw_data_dict['pr_data_best']['px']
                    py = raw_data_dict['pr_data_best'][key_data]
                    names = raw_data_dict['pr_data_best']['names']
                    model = raw_data_dict['validation_best']['model'].split("/")[-1]
                    return px,py,names,model
                
                ## Plot each training result
                colors_list = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
                color_iterator = itertools.cycle(colors_list)
                for index, key in enumerate(self.dataset_train_checkboxes.getChecked()):
                    data = self.dataset_handler[key]
                    if not data:
                        continue
                    try:
                        best_epoch = str(data['train_data']['epoch_best_fit_index'] + 1)
                        px, py, names, model = getLastEpochData(py_tag, data)
                        
                        for i, y in enumerate(py):
                            index_max = len(y)-1
                            # Filter by max recall values so to not have interpolated diagram
                            if canvas_key == 'PR Curve':
                                r = data['pr_data_best']['r'][i]
                                max_r = max(r)
                                index_max = next((i for i, x in enumerate(px) if x > max_r), None)
                            ax.plot(px[:index_max], y[:index_max], linewidth=2, label=f"{self.dataset_handler.getInfo()[key]['name']} ({model}) {names[i]} (best epoch: {best_epoch})", color=next(color_iterator))  # plot(confidence, metric)

                    except KeyError as e:
                        log(f"[{self.__class__.__name__}] Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                        self.dataset_handler.markAsIncomplete(key)

                ## Plot each variance group
                # number_std = self.std_plot_widget.value
                for index, group in enumerate(self.dataset_variance_checkboxes.getChecked()):
                    keys = [key for key in self.dataset_handler.keys() if group in key]
                    py_vec = []
                    for key in keys:
                        data = self.dataset_handler[key]
                        if not data:
                            continue
                        try:
                            px, py, names, model = getLastEpochData(py_tag, data)
                            if len(names) > 1:
                                log(f"[{self.__class__.__name__}] -----------------------------------------------------------------------", bcolors.ERROR)
                                log(f"[{self.__class__.__name__}] - (TDB) MORE THAN ONE TARGET CLASS. NOT BEING HANDLED CORRECTLY (TBD) -", bcolors.ERROR)
                                log(f"[{self.__class__.__name__}] -----------------------------------------------------------------------", bcolors.ERROR)
                            for i, y in enumerate(py):
                                # Filter by max recall values so to not have interpolated diagram. Gets the max index possible
                                # between all recall curves involved in mean and max/min
                                index_max = len(y)-1
                                if canvas_key == 'PR Curve':
                                    r = data['pr_data_best']['r'][i]
                                    max_r = max(r)
                                    index_max = next((i for i, x in enumerate(px) if x > max_r), None)
                                
                                # FIll from index_max to the end with NAN data
                                y = y[:index_max] + [np.nan] * (len(y) - index_max)
                                py_vec.append(y) #[:index_max])

                        except KeyError as e:
                            log(f"[{self.__class__.__name__}] Key error problem generating curve for {key}. It wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                            self.dataset_handler.markAsIncomplete(key)

                    if not py_vec:
                        log(f"[{self.__class__.__name__}] Vector of PY data empty. {group} variance curve wont be plotted.", bcolors.WARNING)
                        continue

                    data_matrix = np.array(py_vec)
                    # nanmean make average ignoring NAN stuff
                    mean = np.nanmean(data_matrix, axis = 0)
                    max_vals = np.nanmax(data_matrix, axis=0)
                    min_vals = np.nanmin(data_matrix, axis=0)

                    # std = np.std(data_matrix, axis = 0, ddof=1)
                    # conf_interval = std*number_std
                    # conf_interval = 1.96*std/math.sqrt(len(keys))
                    # conf_interval = Z(2 STD) * Error típico (la desviación típica que tendría el estadístico si lo calcularas en infinitas muestras iguales)
                    label = group.replace("train_based_variance_", "").replace(".yaml", "")
                    
                    next_color=next(color_iterator)
                    ax.plot(px, mean, label=f"{label}; n = {len(py_vec)}", color=next_color)
                    ax.fill_between(px, min_vals, max_vals, alpha=0.3, facecolor=next_color)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                # self.ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
                ax.set_title(f'{ylabel}-{xlabel} Curve')

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
            
                # Use a Cursor to interactively display the label for a selected line.
                self.cursor[canvas_key] = mplcursors.cursor(ax, hover=True)
                self.cursor[canvas_key].connect("add", lambda sel, xlabel=xlabel, ylabel=ylabel: sel.annotation.set(
                    text=f"{sel.artist.get_label().split(' ')[0]}\n{xlabel}: {sel.target[0]:.2f}, {ylabel}: {sel.target[1]:.2f}",
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgrey', alpha=0.7)
                ))

            # Configurar leyenda
            ax.legend()

        # Actualizar los gráfico
        self.figure_tab_widget.draw()

        # log(f"[{self.__class__.__name__}] Parsing and plot PR-P-R-F1 graphs finished")
