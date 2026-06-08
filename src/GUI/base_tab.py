#!/usr/bin/env python3
# encoding: utf-8
"""
    Defines a Qt tab view with all plot available to compare between different training runs
"""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QAction
from PyQt6.QtWidgets import QHBoxLayout, QWidget, QScrollArea, QSizePolicy, QVBoxLayout, QFileDialog

from utils import parseYaml
from utils import log, bcolors
from .Widgets import PlotTabWidget

class BaseClassPlotter(QWidget):
    # Subclasses that support presets set this to a stable identifier
    # (matches the 'tab' field in the preset YAML).
    TAB_ID = None

    def __init__(self, dataset_handler, tab_keys, tab_id: str = ""):
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
            self.figure_tab_widget = PlotTabWidget(self.tab_keys, tab_id=tab_id)        
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
        
        if hasattr(self, 'figure_tab_widget'):
            self.edit_labels_action = QAction("Edit labels", self)
            self.edit_labels_action.triggered.connect(self.figure_tab_widget.edit_labels)
            edit_menu.addAction(self.edit_labels_action)

            self.edit_xlabels_action = QAction("Edit X labels", self)
            self.edit_xlabels_action.triggered.connect(self.figure_tab_widget.edit_xlabels)
            edit_menu.addAction(self.edit_xlabels_action)

        if hasattr(self, 'csv_tab'):
            self.configure_columns_action = QAction("Configure export columns", self)
            self.configure_columns_action.triggered.connect(self.csv_tab.configure_export_columns)
            edit_menu.addAction(self.configure_columns_action)

        if hasattr(self, 'figure_tab_widget'):
            self.configure_plots_action = QAction("Configure export plots", self)
            self.configure_plots_action.triggered.connect(self.figure_tab_widget.configure_export_plots)
            edit_menu.addAction(self.configure_plots_action)

        # Preset save/load — only registered if the subclass implements the hooks
        if self._supports_presets():
            self.save_preset_action = QAction("Save preset...", self)
            self.save_preset_action.setToolTip("Dump current selection + mode + export filters to a YAML preset")
            self.save_preset_action.triggered.connect(self.save_preset)
            archive_menu.addAction(self.save_preset_action)

            self.load_preset_action = QAction("Load preset...", self)
            self.load_preset_action.setToolTip("Apply a YAML preset to the current view")
            self.load_preset_action.triggered.connect(self.load_preset)
            archive_menu.addAction(self.load_preset_action)

        self.update_checkbox()


    def update_checkbox(self):
        raise NotImplementedError(f"This method has to be reimplemented in child {type(self).__name__} class")

    def save_plot(self):
        raise NotImplementedError(f"This method has to be reimplemented in child {type(self).__name__} class")

    def render_data(self):
        raise NotImplementedError(f"This method has to be reimplemented in child {type(self).__name__} class")

    def _apply_common_suptitle(self, keys):
        """Append a ' - KAIST Day' suffix to every axis title when all plotted
        keys share the same dataset and condition. Silently no-ops when the
        selection is mixed across datasets or conditions.
        """
        from GUI.dataset_manager import extract_common_dataset_condition

        if not hasattr(self, 'figure_tab_widget') or not keys:
            return

        (dataset, condition), is_mixed = extract_common_dataset_condition(
            self.dataset_handler.getInfo(), keys
        )
        if is_mixed or (not dataset and not condition):
            return

        parts = []
        if dataset:
            parts.append(dataset.upper())
        if condition:
            parts.append(condition.title())
        suffix = " - " + " ".join(parts)

        for fig in self.figure_tab_widget.figure.values():
            for ax in fig.axes:
                old = ax.get_title()
                if old and not old.endswith(suffix):
                    ax.set_title(f"{old}{suffix}")

    def save_outputs_to(self, base_path):
        """Save figures (PNG/PDF) and CSV tables to base_path without prompting.

        Used by the headless CLI. Expands ~ and $VARS, creates the parent dir
        if needed. Each consumer applies its own filename suffix and language
        conventions internally.
        """
        base_path = os.path.expanduser(os.path.expandvars(base_path))
        os.makedirs(os.path.dirname(base_path) or '.', exist_ok=True)
        if hasattr(self, 'figure_tab_widget'):
            self.figure_tab_widget.saveFigures(base_path)
        if hasattr(self, 'csv_tab'):
            self.csv_tab.save_data(base_path)

    # ── Preset machinery ──────────────────────────────────────────────
    # Subclasses opt in by setting TAB_ID and implementing the two small
    # _collect_tab_state / _apply_tab_state hooks (only tab-specific bits).
    # Everything common (export filters, filename, dialogs, IO) lives here.

    def _supports_presets(self):
        return bool(self.TAB_ID)

    def _collect_tab_state(self):
        """
        Return the *tab-specific* portion of the preset state:
            {'mode': ..., 'class_key': ..., 'dataset_keys': [...], 'variance_group_keys': [...]}
        Any field can be omitted/None — base will adapt.
        """
        raise NotImplementedError(f"{type(self).__name__} sets TAB_ID but does not implement _collect_tab_state")

    def _apply_tab_state(self, preset):
        """Apply the tab-specific portion (mode + class + selection) from a preset dict."""
        raise NotImplementedError(f"{type(self).__name__} sets TAB_ID but does not implement _apply_tab_state")

    def _collect_preset_state(self):
        from GUI.preset_manager import build_preset
        tab_state = self._collect_tab_state() or {}
        return build_preset(
            tab_id=self.TAB_ID,
            mode=tab_state.get('mode'),
            class_key=tab_state.get('class_key', 'all'),
            dataset_keys=tab_state.get('dataset_keys', []),
            variance_group_keys=tab_state.get('variance_group_keys', []),
            export_plots=self.figure_tab_widget.export_plots if hasattr(self, 'figure_tab_widget') else None,
            export_columns=self.csv_tab.export_columns if hasattr(self, 'csv_tab') else None,
            export_table_formats=self.csv_tab.export_formats if hasattr(self, 'csv_tab') else None,
            export_plot_formats=self.figure_tab_widget.export_formats if hasattr(self, 'figure_tab_widget') else None,
            export_languages=self.csv_tab.export_languages if hasattr(self, 'csv_tab') else None,
            export_filename=getattr(self, '_preset_output_filename', None),
        )

    def _apply_preset_state(self, preset):
        from GUI.preset_manager import plots_for_tab

        # Tab-specific: mode + class + selection
        self._apply_tab_state(preset)

        # Common: export filters
        export = preset.get('export', {}) or {}
        if 'plots' in export and hasattr(self, 'figure_tab_widget'):
            self.figure_tab_widget.export_plots = plots_for_tab(export['plots'], self.TAB_ID)
        if hasattr(self, 'csv_tab'):
            if 'columns' in export:
                self.csv_tab.export_columns = list(export['columns']) or None
            if 'table_formats' in export:
                self.csv_tab.export_formats = list(export['table_formats'])
            if 'languages' in export:
                self.csv_tab.export_languages = list(export['languages']) or None
        if 'plot_formats' in export and hasattr(self, 'figure_tab_widget'):
            self.figure_tab_widget.export_formats = list(export['plot_formats'])
        if 'filename' in export:
            self._preset_output_filename = export['filename'] or None

    def save_preset(self):
        from GUI.preset_manager import save_preset as _save

        file_name, _filter = QFileDialog.getSaveFileName(self, "Save preset", "", "YAML files (*.yaml *.yml);;All Files (*)")
        if not file_name:
            return
        if not file_name.endswith(('.yaml', '.yml')):
            file_name += '.yaml'
        _save(file_name, self._collect_preset_state())

    def load_preset(self):
        from GUI.preset_manager import load_preset as _load

        file_name, _filter = QFileDialog.getOpenFileName(self, "Load preset", "", "YAML files (*.yaml *.yml);;All Files (*)")
        if not file_name:
            return
        preset = _load(file_name)
        if preset is None:
            return
        self._apply_preset_state(preset)
        log(f"[{self.__class__.__name__}] Preset loaded from {file_name}")
