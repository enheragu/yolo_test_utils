#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates a widget with the different tabs that contain graphs as figures
"""
import re
import matplotlib.pyplot as plt
import seaborn as sns

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QWidget, QTabWidget, QVBoxLayout, QDialog, QLineEdit, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from utils import color_palette_list

"""
    Dialog  helper to edit labels of a given plot set
"""
class LabelEditDialog(QDialog):
    def __init__(self, labels):
        super().__init__()
        self.setWindowTitle("Editar Etiquetas")
        self.labels = labels

        layout = QVBoxLayout(self)

        self.line_edits = []
        max_label_width = 0
        for label in labels:
            line_edit = QLineEdit(label)
            max_label_width = max(max_label_width, line_edit.fontMetrics().boundingRect(label).width())
            layout.addWidget(line_edit)
            self.line_edits.append(line_edit)

        for line_edit in self.line_edits:
            line_edit.setMinimumWidth(max_label_width + 20)  # Ajusta el ancho mínimo según el ancho del label

        update_button = QPushButton("Update")
        update_button.clicked.connect(self.accept)
        layout.addWidget(update_button)

    def get_updated_labels(self):
        # Obtener los nuevos nombres de etiquetas desde los QLineEdit
        updated_labels = [line_edit.text() for line_edit in self.line_edits]
        return updated_labels
    
class PlotTabWidget(QTabWidget):
    def __init__(self, tab_keys):
        super().__init__()


        self.tab_canvas = {}
        self.figure = {}
        self.cursor = {}
        
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
        for figure in self.figure.values():
            figure.tight_layout() #(pad = 0.1)
            # figure.subplots_adjust(bottom=0.5, top=0.9)  # Ajusta los márgenes según sea necesario
        
        # plt.tight_layout(pad = 0.1)
        for canvas in self.tab_canvas.values():
            canvas.draw()

    def saveFigures(self, path):
        for key in self.figure.keys():
            self.figure[key].set_size_inches(15, 8)  # Width x Height in inches
            self.figure[key].tight_layout() #(pad = 0.1)
            # self.figure[key].set_size_inches(7.2, 8)  # Width x Height in inches
                        
            plot_name = f"{path}_{key.replace(' ', '_')}.png"
            self.figure[key].savefig(plot_name, format='png')
            print(f"Plot saved to {plot_name}")


    def edit_labels(self):
        labels_dict = {}
        for figure in self.figure.values():
            for ax in figure.ax:
                handles, labels = ax.get_legend_handles_labels()
                labels_dict.update({label: label for label in labels})
        
        if not labels_dict:
            QMessageBox.warning(None, 'No labels found', 'Theres no plot or labels to edit :) Plot something before changing its labels.')
            return
    
        dialog = LabelEditDialog(labels_dict.values())
        if dialog.exec():
            updated_labels = dialog.get_updated_labels()
            print("Updated labels:", updated_labels)
            
            # Actualiza las etiquetas en el diccionario de etiquetas
            for label, item in zip(updated_labels, labels_dict.keys()):
                labels_dict[item] = label
            
            # Actualiza las etiquetas en las leyendas de los gráficos
            for figure in self.figure.values():
                for ax in figure.ax:
                    handles, labels = ax.get_legend_handles_labels()
                    for i, label in enumerate(labels):
                        if label in labels_dict:
                            labels[i] = labels_dict[label]
                    ax.legend(handles, labels)  # Actualiza la leyenda del eje

        # Redibuja los gráficos para aplicar los cambios
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

