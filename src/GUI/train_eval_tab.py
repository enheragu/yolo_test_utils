#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt tab view with all plot available to compare between different training runs
"""
from PyQt6.QtGui import QKeySequence, QAction
from PyQt6.QtWidgets import QPushButton, QFileDialog, QSizePolicy

from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns

from utils import log, bcolors
from GUI.gui_config import get_train_eval_tab_keys
from GUI.base_tab import BaseClassPlotter
from GUI.Widgets import DatasetCheckBoxWidget, DialogWithCheckbox, BestGroupCheckBoxWidget

# Tab keys loaded from centralized config (config/features.py)
tab_keys = get_train_eval_tab_keys()

class TrainEvalPlotter(BaseClassPlotter):
    def __init__(self, dataset_handler):
        super().__init__(dataset_handler, tab_keys, tab_id="train_eval")

        self.dataset_checkboxes = DatasetCheckBoxWidget(self.options_widget, dataset_handler, title_filter=["train_based_"])
        self.options_layout.insertWidget(0, self.dataset_checkboxes,3)

        ## --- Extra selector (created lazily to avoid ~6s startup cost of 502 checkboxes)
        self._dataset_checkboxes_extra = None
        self._extra_dataset_dialog = None

        self.dataset_variance_checkboxes = BestGroupCheckBoxWidget(self.options_widget, dataset_handler, include="variance_", title="(Best) Variance analysis sets:", title_filter=["variance_"], class_selector=None)
        self.options_layout.insertWidget(0, self.dataset_variance_checkboxes,3)

        # self.select_extra_button.clicked.connect(self.extra_dataset_dialog.show)
        ## ---

        self.select_all_button = QPushButton(" Select All ", self)
        self.select_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.select_all_button.clicked.connect(lambda: (self.dataset_checkboxes.select_all(), self._select_all_extra(), self.dataset_variance_checkboxes.select_all()))

        self.deselect_all_button = QPushButton(" Deselect All ", self)
        self.deselect_all_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.deselect_all_button.clicked.connect(lambda: (self.dataset_checkboxes.deselect_all(), self._deselect_all_extra(), self.dataset_variance_checkboxes.deselect_all()))

        self.plot_button = QPushButton(" Generate Plot ", self)
        self.plot_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_button.clicked.connect(self.render_data)

        self.save_button = QPushButton(" Save Output ", self)
        self.save_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.save_button.clicked.connect(self.save_plot)

        self.buttons_layout.addWidget(self.select_all_button)
        self.buttons_layout.addWidget(self.deselect_all_button)
        self.buttons_layout.addWidget(self.plot_button)
        self.buttons_layout.addWidget(self.save_button)
        # self.buttons_layout.addWidget(self.select_extra_button)   
        
        log(f"[{self.__class__.__name__}] Initialized.")

    def update_view_and_menu(self, menu_list):
        archive_menu, view_menu, tools_menu, edit_menu = menu_list

        self.selec_extra_action = QAction("Select extra data", self)
        self.selec_extra_action.setToolTip('Allows to choose single variance tests instead of plotting them as a group')
        self.selec_extra_action.triggered.connect(self.extra_dataset_dialog.show)
        tools_menu.addAction(self.selec_extra_action)
        
        super().update_view_and_menu(menu_list)

    @property
    def dataset_checkboxes_extra(self):
        """Lazy-create the extra checkbox widget (502 checkboxes) on first access."""
        if self._dataset_checkboxes_extra is None:
            import time as _time
            t0 = _time.time()
            self._dataset_checkboxes_extra = DatasetCheckBoxWidget(
                self.options_widget, self.dataset_handler, exclude=None,
                include="variance_", title_filter=["train_based_"], max_rows=8)
            self._dataset_checkboxes_extra.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            log(f"[{self.__class__.__name__}] Extra checkbox widget created lazily in {_time.time() - t0:.2f}s.")
        return self._dataset_checkboxes_extra

    @property
    def extra_dataset_dialog(self):
        """Lazy-create the extra dataset dialog on first access."""
        if self._extra_dataset_dialog is None:
            self._extra_dataset_dialog = DialogWithCheckbox(
                title="Extra dataset selector",
                checkbox_widget=self.dataset_checkboxes_extra,
                render_func=self.render_data)
        return self._extra_dataset_dialog

    def _select_all_extra(self):
        if self._dataset_checkboxes_extra is not None:
            self._dataset_checkboxes_extra.select_all()

    def _deselect_all_extra(self):
        if self._dataset_checkboxes_extra is not None:
            self._dataset_checkboxes_extra.deselect_all()

    def update_checkbox(self):
        self.dataset_checkboxes.update_checkboxes()
        if self._dataset_checkboxes_extra is not None:
            self._dataset_checkboxes_extra.update_checkboxes()
        self.dataset_variance_checkboxes.update_checkboxes()

    def save_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plots as PNG Images", "", "PNG Images (*.png);;All Files (*)")

        if file_name:
            self.figure_tab_widget.saveFigures(file_name)

    def render_data(self):
        extra_checked = self.dataset_checkboxes_extra.getChecked() if self._dataset_checkboxes_extra is not None else []
        checked = extra_checked + self.dataset_checkboxes.getChecked() + self.dataset_variance_checkboxes.getChecked()
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
                        # subplot[py].plot(data_x, data_y, marker='.', label=key, linewidth=4, markersize=10)  # actual results
                        # subplot[py].plot(data_x, gaussian_filter1d(data_y, sigma=3), ':', label=f'{key}-smooth', linewidth=4)  # smoothing line
                        # subplot[py].plot(data_x, data_y, linewidth=1)  # plot(confidence, metric)

                        title = self.dataset_handler.getInfo()[key]['label']
                        sns.lineplot(x=data_x, y=data_y, marker='.', label=title, linewidth=4, markersize=10, ax = subplot[py])
                        smoothed_data_y = gaussian_filter1d(data_y, sigma=3)
                        sns.lineplot(x=data_x, y=smoothed_data_y, linestyle=':', label=f'{title} (smooth)', linewidth=4, ax = subplot[py])

                        # Configurar leyenda
                        subplot[py].set_xlabel("epoch")
                        # subplot[py].set_ylabel(py)
                        subplot[py].legend()
                    except KeyError as e:
                        log(f"[{self.__class__.__name__}] Key error problem generating Train/Val plots for {key}. Row wont be generated. Missing key in data dict: {e}", bcolors.ERROR)
                        self.dataset_handler.markAsIncomplete(key)

                self.figure_tab_widget[canvas_key].ax.append(subplot[py])

            self.figure_tab_widget[canvas_key].subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
            self.figure_tab_widget[canvas_key].tight_layout()

        self._apply_common_suptitle(checked_list)

        # Actualizar los gráfico
        self.figure_tab_widget.draw()
        # log(f"[{self.__class__.__name__}] Parsing and plot Loss Val PR and mAP graphs finished")

