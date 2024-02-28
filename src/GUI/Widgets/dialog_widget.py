#!/usr/bin/env python3
# encoding: utf-8

"""
    Defines a new pop up dialog that contains a given widget. It
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QDialog, QVBoxLayout

class DialogWithCheckbox(QDialog):
    def __init__(self, title, checkbox_widget, render_func):
        super().__init__()
        self.setWindowTitle(title)
        self.render_data_callback = render_func
        layout = QVBoxLayout()
        layout.addWidget(checkbox_widget)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply)
        layout.addWidget(apply_button)

        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(checkbox_widget.select_all)
        layout.addWidget(select_all_button)

        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(checkbox_widget.deselect_all)
        layout.addWidget(deselect_all_button)


        self.setLayout(layout)
    
    def apply(self):
        self.hide()
        self.render_data_callback()