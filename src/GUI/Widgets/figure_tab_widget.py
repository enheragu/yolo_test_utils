#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates a widget with the different tabs that contain graphs as figures
"""
import re
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMessageBox, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QDialog, QLineEdit, QPushButton, QLabel, QGridLayout, QGroupBox,
    QListWidget, QListWidgetItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from utils import color_palette_list
from GUI.gui_config import (
    load_label_mappings, save_label_mappings,
    load_plot_selection, save_plot_selection,
    get_export_languages, get_language_suffix, get_plot_labels, set_language, get_language
)

"""
    Dialog helper to edit labels of a given plot set.
    When export mode is 'all', shows one column per language (EN / ES).
    Otherwise shows a single editable column.

    label_mappings format: {original: {lang: display, ...}}
"""
class LabelEditDialog(QDialog):
    def __init__(self, original_labels, label_mappings=None, languages=None):
        """
        Args:
            original_labels: List of original label names (read-only).
            label_mappings: Dict {original: {lang: display, ...}}.  None → identity.
            languages: List of language codes to show columns for (e.g. ['en'] or ['en','es']).
        """
        super().__init__()
        self.setWindowTitle("Edit Labels")
        self.original_labels = list(original_labels)
        self.languages = languages or ['en']

        if label_mappings is None:
            label_mappings = {}

        multi = len(self.languages) > 1

        main_layout = QVBoxLayout(self)

        # ── Header row ──
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<b>Original</b>"), 1)
        header_layout.addWidget(QLabel("<b>→</b>"))
        if multi:
            for lang in self.languages:
                header_layout.addWidget(QLabel(f"<b>{lang.upper()}</b>"), 1)
        else:
            header_layout.addWidget(QLabel("<b>New Name</b>"), 1)
        main_layout.addLayout(header_layout)

        # ── Rows: one per label, one QLineEdit per language ──
        # self.line_edits[i] = {lang: QLineEdit}
        self.line_edits: list[dict[str, QLineEdit]] = []
        max_label_width = 0

        for orig in self.original_labels:
            row_layout = QHBoxLayout()

            # Original label (read-only)
            orig_widget = QLabel(orig)
            orig_widget.setStyleSheet("color: gray; font-style: italic;")
            max_label_width = max(max_label_width, orig_widget.fontMetrics().boundingRect(orig).width())
            row_layout.addWidget(orig_widget, 1)
            row_layout.addWidget(QLabel("→"))

            lang_edits: dict[str, QLineEdit] = {}
            mapping = label_mappings.get(orig, {})
            for lang in self.languages:
                display = mapping.get(lang, orig) if isinstance(mapping, dict) else orig
                le = QLineEdit(display)
                le.setPlaceholderText(orig)
                row_layout.addWidget(le, 1)
                lang_edits[lang] = le

            self.line_edits.append(lang_edits)
            main_layout.addLayout(row_layout)

        # Minimum widths
        min_w = max(200, max_label_width + 20)
        for lang_edits in self.line_edits:
            for le in lang_edits.values():
                le.setMinimumWidth(min_w)

        # ── Buttons ──
        button_layout = QHBoxLayout()

        reset_button = QPushButton("Reset to Original")
        reset_button.clicked.connect(self._reset_labels)
        button_layout.addWidget(reset_button)

        update_button = QPushButton("Apply")
        update_button.clicked.connect(self.accept)
        update_button.setDefault(True)
        button_layout.addWidget(update_button)

        main_layout.addLayout(button_layout)
        self.setMinimumWidth(400 if not multi else 650)

    def _reset_labels(self):
        """Reset all display labels to their original values."""
        for lang_edits, orig in zip(self.line_edits, self.original_labels):
            for le in lang_edits.values():
                le.setText(orig)

    def get_updated_labels(self) -> dict:
        """
        Return {original: {lang: display, ...}} for every label.
        """
        result = {}
        for orig, lang_edits in zip(self.original_labels, self.line_edits):
            result[orig] = {lang: le.text() for lang, le in lang_edits.items()}
        return result


class PlotSelectorDialog(QDialog):
    """
    Dialog to select which plots to include when exporting.
    Simple checklist — no drag-drop reordering needed.
    """
    def __init__(self, plot_keys: list, selected: Optional[list] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Export Plots")
        self.setMinimumSize(350, 400)

        self.all_keys = plot_keys
        if selected is None:
            selected = plot_keys  # all selected by default

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Check plots to include in export:"))

        self.list_widget = QListWidget()
        for key in plot_keys:
            item = QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if key in selected else Qt.CheckState.Unchecked
            )
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        # Select / Deselect buttons
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        btn_layout.addWidget(deselect_all_btn)
        layout.addLayout(btn_layout)

        # Apply / Cancel
        action_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        action_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.accept)
        apply_btn.setDefault(True)
        action_layout.addWidget(apply_btn)
        layout.addLayout(action_layout)

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)

    def get_selected_plots(self) -> list:
        """Return list of checked plot keys."""
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected


class PlotTabWidget(QTabWidget):
    def __init__(self, tab_keys, tab_id: str = ""):
        super().__init__()

        self.tab_canvas = {}
        self.figure = {}
        self.cursor = {}
        self.label_mappings = load_label_mappings()  # Load from session cache
        self.tab_id = tab_id  # Used as key for persisting plot selection
        self.export_plots: Optional[list] = load_plot_selection(tab_id) if tab_id else None
        self.export_formats: Optional[list] = None  # None = default ['png']; set to filter (e.g. ['pdf'])
        
        plt.rc('font', size=14)

        # Crear pestañas para cada conjunto de datos
        for key in tab_keys:
            # Create a Matplotlib widget
            self.figure[key] = plt.figure()
            self.figure[key].ax = []
            ax = self.figure[key].add_subplot(111)
            ax.axis('off')
            ax.text(0.5,0.5, 'Select datasets to plot', ha='center', va='center', fontsize=28, color='gray')
            self.tab_canvas[key] = FigureCanvas(self.figure[key])
            
            toolbar = NavigationToolbar(self.tab_canvas[key])
            
            # Agregar el lienzo a la pestaña
            tab = QWidget()
            tab.layout = QVBoxLayout()
            tab.layout.addWidget(toolbar)
            tab.layout.addWidget(self.tab_canvas[key] )
            tab.setLayout(tab.layout)
            self.addTab(tab, key)

        # sns.set_palette("colorblind")
        sns.set_palette(sns.color_palette(color_palette_list))

    def __getitem__(self, key):
        return self.figure[key]
    
    def clear(self):
        for figure in self.figure.values():
            if hasattr(figure, "ax") and isinstance(figure.ax, list) and figure.ax:
                figure.ax = []
            figure.clear()
        for canvas in self.tab_canvas.values():
            canvas.draw()

    def draw(self):
        # Apply any stored label mappings before drawing
        self.apply_label_mappings()
        
        for figure in self.figure.values():
            figure.tight_layout() #(pad = 0.1)
            # figure.subplots_adjust(bottom=0.5, top=0.9)  # Ajusta los márgenes según sea necesario
        
        # plt.tight_layout(pad = 0.1)
        for canvas in self.tab_canvas.values():
            canvas.draw()

    def configure_export_plots(self):
        """Open dialog to choose which plots will be exported."""
        all_keys = list(self.figure.keys())
        dialog = PlotSelectorDialog(all_keys, self.export_plots, parent=self)
        if dialog.exec():
            selected = dialog.get_selected_plots()
            # None means "all" (no filter)
            self.export_plots = selected if len(selected) < len(all_keys) else None
            if self.tab_id:
                save_plot_selection(self.tab_id, self.export_plots)
            print(f"Export plots: {self.export_plots or 'all'}")

    def saveFigures(self, path):
        """
        Save selected figures to image files (PNG and/or PDF).
        If export mode is 'all', saves in all languages with suffixes.
        Respects self.export_plots selection (None = export all).
        Respects self.export_formats (None = default ['png']; e.g. ['pdf'] for vector output).
        """
        formats = list(self.export_formats) if self.export_formats is not None else ['png']
        formats = [f for f in formats if f in ('png', 'pdf')]
        if not formats:
            return  # nothing to save

        export_langs = get_export_languages()
        original_lang = get_language()

        keys_to_export = (
            [k for k in self.figure if k in self.export_plots]
            if self.export_plots is not None
            else list(self.figure.keys())
        )

        for lang in export_langs:
            set_language(lang)
            lang_suffix = get_language_suffix(lang)

            self._apply_language_to_axes(lang)
            self.apply_label_mappings(lang)

            for key in keys_to_export:
                self.figure[key].set_size_inches(8, 8)
                self.figure[key].tight_layout()

                for fmt in formats:
                    plot_name = f"{path}_{key.replace(' ', '_')}{lang_suffix}.{fmt}"
                    self.figure[key].savefig(plot_name, format=fmt)
                    print(f"Plot saved to {plot_name}")
        
        # Restore original language
        set_language(original_lang)
        if len(export_langs) > 1:
            self._apply_language_to_axes(original_lang)
            self.apply_label_mappings(original_lang)
    
    def _apply_language_to_axes(self, lang: str):
        """Apply language-specific translations to axis labels."""
        labels = get_plot_labels(lang)
        
        for figure in self.figure.values():
            for ax in figure.ax:
                # Update xlabel if translatable
                current_xlabel = ax.get_xlabel()
                if current_xlabel in labels['xlabel']:
                    ax.set_xlabel(labels['xlabel'][current_xlabel])
                
                # Update ylabel if translatable
                current_ylabel = ax.get_ylabel()
                if current_ylabel in labels['ylabel']:
                    ax.set_ylabel(labels['ylabel'][current_ylabel])

    def apply_label_mappings(self, lang: Optional[str] = None):
        """
        Apply stored label mappings to all figure legends.
        label_mappings format: {original: {lang: display, ...}}.
        If lang is None, uses get_language().
        """
        if not self.label_mappings:
            return

        use_lang = lang or get_language()

        for figure in self.figure.values():
            for ax in figure.ax:
                handles, labels = ax.get_legend_handles_labels()
                updated_labels = []
                for label in labels:
                    mapping = self.label_mappings.get(label)
                    if isinstance(mapping, dict):
                        updated_labels.append(mapping.get(use_lang, label))
                    else:
                        updated_labels.append(label)
                if labels != updated_labels:
                    ax.legend(handles, updated_labels)

    def edit_labels(self):
        # Collect all original labels from current plots
        original_labels = []
        seen = set()
        for figure in self.figure.values():
            for ax in figure.ax:
                handles, labels = ax.get_legend_handles_labels()
                for label in labels:
                    if label not in seen:
                        seen.add(label)
                        original_labels.append(label)
        
        if not original_labels:
            QMessageBox.warning(None, 'No labels found', 'There are no plots or labels to edit. Plot something first.')
            return
        
        export_langs = get_export_languages()

        dialog = LabelEditDialog(original_labels, self.label_mappings, languages=export_langs)
        if dialog.exec():
            # Update persistent mappings  {orig: {lang: display}}
            new_mappings = dialog.get_updated_labels()
            self.label_mappings.update(new_mappings)
            
            # Remove identity mappings (all lang values == key)
            self.label_mappings = {
                k: v for k, v in self.label_mappings.items()
                if isinstance(v, dict) and any(lv != k for lv in v.values())
            }
            
            # Save to cache for persistence between sessions
            save_label_mappings(self.label_mappings)
            
            print("Label mappings:", self.label_mappings)
            
            # Apply mappings to current plots
            self.apply_label_mappings()

        # Redraw to apply changes
        self.draw()

    def edit_xlabels(self):
        xlabels_dict = {}
        rotation_dict = {}
        alignment_dict = {}


        def is_numeric_label(label):
            """Comprueba si una etiqueta es numérica usando expresiones regulares."""
            return bool(re.match(r"^\d+(\.\d+)?$", label))

        # Recolecta todas las etiquetas del eje X y sus propiedades de rotación y alineación
        for figure in self.figure.values():
            for ax in figure.ax:
                xlabels = [label.get_text() for label in ax.get_xticklabels()]
                xrotations = [label.get_rotation() for label in ax.get_xticklabels()]
                xalignments = [label.get_ha() for label in ax.get_xticklabels()]  # 'ha' = horizontal alignment

                for label, rotation, alignment in zip(xlabels, xrotations, xalignments):
                    if not is_numeric_label(label):  # Solo procesa etiquetas no numéricas
                        xlabels_dict[label] = label
                        rotation_dict[label] = rotation
                        alignment_dict[label] = alignment
        
        if not xlabels_dict:
            QMessageBox.warning(None, 'No labels found', 'Theres no plot or labels to edit :) Plot something before changing its labels.')
            return

        # Lanza un diálogo para editar las etiquetas del eje X
        dialog = LabelEditDialog(xlabels_dict.values())
        if dialog.exec():
            updated_xlabels = dialog.get_updated_labels()
            print("Updated X labels:", updated_xlabels)
            
            # Actualiza las etiquetas en el diccionario de etiquetas del eje X
            for label, item in zip(updated_xlabels, xlabels_dict.keys()):
                xlabels_dict[item] = label
            
            # Actualiza las etiquetas en los ejes X de los gráficos
            for figure in self.figure.values():
                for ax in figure.ax:
                    current_xlabels = [label.get_text() for label in ax.get_xticklabels()]
                    new_xlabels = [xlabels_dict[label] if label in xlabels_dict else label for label in current_xlabels]
                    
                    # Aplica las etiquetas nuevas, conservando la rotación y alineación previas
                    ax.set_xticklabels(new_xlabels)
                    for label in ax.get_xticklabels():
                        label_text = label.get_text()
                        if label_text in rotation_dict:
                            label.set_rotation(rotation_dict[label_text])
                        if label_text in alignment_dict:
                            label.set_ha(alignment_dict[label_text])  # Aplica la alineación horizontal anterior

        # Redibuja los gráficos para aplicar los cambios
        self.draw()

