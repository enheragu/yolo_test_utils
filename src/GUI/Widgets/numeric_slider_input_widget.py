#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates a widget with a title, a numeric input box and an associated slider
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QLabel, QSlider

class NumericSliderInputWidget(QWidget):
    def __init__(self, label_text='', min = 0, max = 4, parent=None):
        super().__init__(parent)

        self.value = 0.0
        self.minimum = 0.0
        self.maximum = 5.0
        self.decimals = 2

        layout = QHBoxLayout()
        self.setLayout(layout)

        # Label para la descripción del input
        self.label = QLabel(label_text)
        layout.addWidget(self.label)


        # Slider para ajustar el valor
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.valueChanged.connect(self.update_value_from_slider)

        # Campo de texto para ingresar el valor numérico
        self.input_field = QLineEdit()
        self.input_field.textChanged.connect(self.update_value_from_input)
        self.input_field.setText('2.00')
        self.input_field.setMaximumWidth(100)  # Establecer el ancho máximo

        layout.addWidget(self.input_field)
        layout.addWidget(self.slider)

    def update_value_from_slider(self):
        self.value = self.slider.value() * (self.maximum - self.minimum) / 100 + self.minimum
        self.input_field.setText(f'{self.value:.{self.decimals}f}')

    def update_value_from_input(self):
        text = self.input_field.text()
        try:
            self.value = float(text)
            if self.value < self.minimum:
                self.value = self.minimum
            elif self.value > self.maximum:
                self.value = self.maximum
        except ValueError:
            pass
        self.slider.setValue(int((self.value - self.minimum) * 100 / (self.maximum - self.minimum)))

