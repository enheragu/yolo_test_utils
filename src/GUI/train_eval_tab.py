#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt tab view with all plot available to compare between different training runs
"""

from PyQt6.QtWidgets import QPushButton, QFileDialog, QSizePolicy

from scipy.ndimage.filters import gaussian_filter1d

from log_utils import log, bcolors
from GUI.base_tab import BaseClassPlotter
from GUI.Widgets import DatasetCheckBoxWidget, DialogWithCheckbox

tab_keys = ['Train Loss Ev.', 'Val Loss Ev.', 'PR Evolution', 'mAP Evolution' ]

class TrainEvalPlotter(BaseClassPlotter):
    def __init__(self, dataset_handler):
        super().__init__(dataset_handler, tab_keys)

        self.dataset_checkboxes = DatasetCheckBoxWidget(self.options_widget, dataset_handler, title_filter=["train_based_"])
        self.options_layout.insertWidget(0, self.dataset_checkboxes,3)

        ## --- Adds window selector to be able to add manually individual tests from variance_ stuff
        self.dataset_checkboxes_extra = DatasetCheckBoxWidget(self.options_widget, dataset_handler, exclude = None, include="variance_", title_filter=["train_based_"], max_rows = 8)
        self.dataset_checkboxes_extra.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.select_extra_button = QPushButton(" Select extra ")
        self.select_extra_button.setToolTip('Allows to choose single variance tests instead of plotting them as a group')
        self.select_extra_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.extra_dataset_dialog = DialogWithCheckbox(title="Extra dataset selector", checkbox_widget=self.dataset_checkboxes_extra, render_func = self.render_data)
        self.select_extra_button.clicked.connect(self.extra_dataset_dialog.show)
        ## ---

        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.deselect_all_button.clicked.connect(lambda: (self.dataset_checkboxes.deselect_all(), self.dataset_checkboxes_extra.deselect_all()))

        self.plot_button = QPushButton(" Generate Plot ", self)
        self.plot_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_button.clicked.connect(self.render_data)

        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.save_button.clicked.connect(self.save_plot)

        self.buttons_layout.addWidget(self.deselect_all_button)
        self.buttons_layout.addWidget(self.plot_button)
        self.buttons_layout.addWidget(self.save_button)
        self.buttons_layout.addWidget(self.select_extra_button)
   
    def update_checkbox(self):
        self.dataset_checkboxes.update_checkboxes()
        self.dataset_checkboxes_extra.update_checkboxes()

    def save_plot(self):
        # Open a file dialog to select the saving location
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)")

        if file_name:
            self.figure_tab_widget.saveFigures(file_name)

    def render_data(self):
        checked = self.dataset_checkboxes_extra.getChecked() + self.dataset_checkboxes.getChecked()
        self.plot_loss_metrics_data(checked)
        log(f"[{self.__class__.__name__}] Plot updated")

    def plot_loss_metrics_data(self, checked_list):

        plot_data = {'Train Loss Ev.': {'py': ['train/box_loss', 'train/cls_loss', 'train/dfl_loss'], 'xlabel': "epoch"},
                     'Val Loss Ev.': {'py': ['val/box_loss', 'val/cls_loss', 'val/dfl_loss'], 'xlabel': "epoch"},
                     'PR Evolution': {'py': ['metrics/precision(B)', 'metrics/recall(B)'], 'xlabel': "epoch"},
                     'mAP Evolution': {'py': ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'], 'xlabel': "epoch"}}
        
        # Borrar gráficos previos
        self.figure_tab_widget.clear()

        for canvas_key in self.tab_keys:
            if canvas_key not in plot_data:
                continue

            subplot = {}
            for index, py in enumerate(plot_data[canvas_key]['py']):
                subplot[py] = self.figure_tab_widget[canvas_key].add_subplot(1, len(plot_data[canvas_key]['py']), index+1) # index + 1 as 0 is not allowed
                subplot[py].set_title(f'{py}')
                for key in checked_list:
                    try:
                        if 'csv_data' not in self.dataset_handler[key]:
                            log(f"No CSV data associated to {key}", bcolors.ERROR)
                            continue

                        # Is done at the end of plotting, so data should already be in parsed dict             
                        data = self.dataset_handler[key]['csv_data']
                        data_x = [int(x) for x in data['epoch']]
                        data_y = [float(y) for y in data[py]]
                        subplot[py].plot(data_x, data_y, marker='.', label=key, linewidth=4, markersize=10)  # actual results
                        subplot[py].plot(data_x, gaussian_filter1d(data_y, sigma=3), ':', label=f'{key}-smooth', linewidth=4)  # smoothing line
                        # subplot[py].plot(data_x, data_y, linewidth=1)  # plot(confidence, metric)

                        # Configurar leyenda
                        subplot[py].set_xlabel("epoch")
                        # subplot[py].set_ylabel(py)
                        subplot[py].legend()
                    except KeyError as e:
                        log(f"[{self.__class__.__name__}] Key error problem generating Train/Val plots for {key}. Row wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                        self.dataset_handler.markAsIncomplete(key)
            
            self.figure_tab_widget[canvas_key].subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
            self.figure_tab_widget[canvas_key].tight_layout()

        # Actualizar los gráfico
        self.figure_tab_widget.draw()
        # log(f"[{self.__class__.__name__}] Parsing and plot Loss Val PR and mAP graphs finished")

