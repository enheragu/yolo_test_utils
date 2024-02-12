#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates a widget with slide view that contains all checkboxes with dataset parsed
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QWidget, QCheckBox, QGroupBox, QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QTableWidget, QTableWidgetItem

max_rows_checkboxes = 5


class DatasetCheckBoxWidget(QScrollArea):
    """
        :param: include  Str pattern of tests that will be included in the checbox widget
                as 'train_' or 'variance_'
    """
    def __init__(self, widget, dataset_handler, include = None, exclude = "variance_", max_rows = max_rows_checkboxes):
        super().__init__(widget)

        self.setWidgetResizable(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Hacerlo redimensionable solo horizontalmente

        # Crear un widget que contendrá los grupos de checkboxes
        scroll_widget = QWidget()
        scroll_widget.setMinimumHeight(max_rows) # por lo que sea poner un mínimo hace que evite el slider vertical lateral...
        self.setWidget(scroll_widget)
        scroll_layout = QGridLayout(scroll_widget)

        # Create group boxes to group checkboxes
        group_dict = {}
        last_group = ""
        iter = 0
        self.check_box_dict = {}

        for key, dataset_info in dataset_handler.getInfo().items():
            group_name = dataset_info['model']
            if (include and include not in group_name) or \
               (exclude and exclude in group_name): 
                continue

            test_name = dataset_info['name']
            if group_name != last_group:
                iter = 0
                last_group = group_name
                group_dict[group_name] = QGroupBox(f"Model: {group_name}")
                group_dict[group_name].setLayout(QGridLayout())
                group_dict[group_name].setStyleSheet("font-weight: bold;")
                scroll_layout.addWidget(group_dict[group_name], 0, len(group_dict) - 1)

            checkbox = QCheckBox(test_name)
            checkbox.setStyleSheet("font-weight: normal;") # Undo the bold text from parent 
            self.check_box_dict[key] = checkbox
            row = iter % max_rows
            col = iter // max_rows
            group_dict[group_name].layout().addWidget(checkbox, row, col)
            iter += 1
            
    def getChecked(self):
        return [key for key, checkbox in self.check_box_dict.items() if checkbox.isChecked()]
        
    def isChecked(self, key):
        if key not in self.check_box_dict:
            return False
        return self.check_box_dict[key].isChecked()

    def select_all(self):
        for checkbox in self.check_box_dict.values():
            checkbox.setChecked(True)

    def select_cond(self, condition):
        for key, checkbox in self.check_box_dict.items():
            if condition in key:
                checkbox.setChecked(True)

    def deselect_all(self):
        for checkbox in self.check_box_dict.values():
            checkbox.setChecked(False)




class GroupCheckBoxWidget(QScrollArea):
    """
        :param: include  Str pattern of tests that will be included in the checbox widget
                as 'train_' or 'variance_'
    """
    def __init__(self, widget, dataset_handler, include = None, exclude = None, title = "", max_rows = max_rows_checkboxes):
        super().__init__(widget)

        self.setWidgetResizable(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Hacerlo redimensionable solo horizontalmente

        # Crear un widget que contendrá los grupos de checkboxes
        scroll_widget = QWidget()
        scroll_widget.setMinimumHeight(max_rows) # por lo que sea poner un mínimo hace que evite el slider vertical lateral...
        self.setWidget(scroll_widget)
        scroll_layout = QGridLayout(scroll_widget)

        # Create group boxes to group checkboxes
        group_dict = {}
        last_group = ""
        iter = 0
        self.check_box_dict = {}

        LabelGroup = QGroupBox(title)
        LabelGroup.setLayout(QGridLayout())
        LabelGroup.setStyleSheet("font-weight: bold;")
        scroll_layout.addWidget(LabelGroup)
        
        for key, dataset_info in dataset_handler.getInfo().items():
            group_name = dataset_info['model']
            if not group_name in self.check_box_dict:
                if (include and include not in group_name) or \
                (exclude and exclude in group_name): 
                    continue
                    
                checkbox = QCheckBox(group_name)
                checkbox.setStyleSheet("font-weight: normal;") # Undo the bold text from parent 
                self.check_box_dict[group_name] = checkbox
                row = iter % max_rows
                col = iter // max_rows
                LabelGroup.layout().addWidget(checkbox, row, col)
                
                iter += 1
                
    def getChecked(self):
        return [key for key, checkbox in self.check_box_dict.items() if checkbox.isChecked()]
        
    def isChecked(self, key):
        key_group = key.split("/")[0]
        if key_group not in self.check_box_dict:
            return False
        return self.check_box_dict[key_group].isChecked()

    def select_all(self):
        for checkbox in self.check_box_dict.values():
            checkbox.setChecked(True)

    def select_cond(self, condition):
        for key, checkbox in self.check_box_dict.items():
            if condition in key:
                checkbox.setChecked(True)

    def deselect_all(self):
        for checkbox in self.check_box_dict.values():
            checkbox.setChecked(False)