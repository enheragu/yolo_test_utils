#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates a widget with the different tabs that contain graphs as figures
"""

import matplotlib.pyplot as plt

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QDialog, QLineEdit, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

        # Crear pestañas para cada conjunto de datos
        for key in tab_keys:
            # Create a Matplotlib widget
            self.figure[key] = plt.figure()
            self.figure[key].ax = []
            ax = self.figure[key].add_subplot(111)
            ax.axis('off')
            ax.text(0.5,0.5, 'Select datasets to plot', ha='center', va='center', fontsize=28, color='gray')
            self.tab_canvas[key] = FigureCanvas(self.figure[key])
            
            # Agregar el lienzo a la pestaña
            tab = QWidget()
            tab.layout = QVBoxLayout()
            tab.layout.addWidget(self.tab_canvas[key] )
            tab.setLayout(tab.layout)
            self.addTab(tab, key)

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
        # for figure in self.figure.values():
            # figure.tight_layout()
        for canvas in self.tab_canvas.values():
            canvas.draw()

    def saveFigures(self, path):
        for key in self.figure.keys():
            # Save the plot to the selected location as a PNG image
            plot_name = path + "_" + key.replace(" ", "_") + ".pdf" #".png"
            self.figure[key].savefig(plot_name)#, format='png')
            print(f"Plot saved to {plot_name}.png")
        
    def edit_labels(self):
        labels_dict = {}
        for figure in self.figure.values():
            for ax in figure.ax:
                handles, labels = ax.get_legend_handles_labels()
                labels_dict.update({label: label for label in labels})
        
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
