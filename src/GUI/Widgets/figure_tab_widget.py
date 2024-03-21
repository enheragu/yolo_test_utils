#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates a widget with the different tabs that contain graphs as figures
"""

import matplotlib.pyplot as plt

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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