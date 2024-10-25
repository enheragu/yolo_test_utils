#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates a widget with slide view that contains all checkboxes with dataset parsed
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QWidget, QCheckBox, QGroupBox, QScrollArea, QSizePolicy, QDialog

max_rows_checkboxes = 4

# Remove machine tag
title_filter_global = []
class DatasetCheckBoxWidget(QScrollArea):
    """
        :param: include  Str pattern of tests that will be included in the checbox widget
                as 'train_' or 'variance_'
        :param: title_filter: list of substrings to filter from Group title
    """
    def __init__(self, widget, dataset_handler, include = None, exclude = "variance_", max_rows = max_rows_checkboxes, title_filter = []):
        global title_filter_global
        super().__init__(widget)

        self.dataset_handler = dataset_handler
        self.include = include
        self.exclude = exclude
        self.title_filter = title_filter + title_filter_global
        self.max_rows = max_rows

        self.setWidgetResizable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Hacerlo redimensionable solo horizontalmente

        # Crear un widget que contendrá los grupos de checkboxes
        self.scroll_widget = QWidget()
        self.scroll_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_widget.setMinimumHeight(max_rows) # por lo que sea poner un mínimo hace que evite el slider vertical lateral...
        self.setWidget(self.scroll_widget)
        self.scroll_layout = QGridLayout(self.scroll_widget)

        # Create group boxes to group checkboxes
        self.check_box_dict = {}
        self.update_checkboxes()

    def update_checkboxes(self):
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            
        self.check_box_dict = {}
        group_dict = {}
        last_group = ""
        iter = 0
        for key, dataset_info in self.dataset_handler.getInfo().items():
            group_name = dataset_info['model']
            if (self.include and self.include not in group_name) or \
               (self.exclude and self.exclude in group_name): 
                continue

            test_name = dataset_info['title']
            if group_name != last_group:
                iter = 0
                last_group = group_name
                title_name = group_name
                for filter in self.title_filter:
                    title_name = title_name.replace(filter, "")
                group_dict[group_name] = QGroupBox(f"Model: {title_name}")
                group_dict[group_name].setLayout(QGridLayout())
                group_dict[group_name].setStyleSheet("font-weight: bold;")
                self.scroll_layout.addWidget(group_dict[group_name], 0, len(group_dict) - 1)

            for filter in self.title_filter:
                test_name = test_name.replace(filter, "")
            checkbox = QCheckBox(test_name)
            checkbox.setToolTip(dataset_info['path'])
            checkbox.setStyleSheet("font-weight: normal;") # Undo the bold text from parent 
            self.check_box_dict[key] = checkbox
            row = iter % self.max_rows
            col = iter // self.max_rows
            group_dict[group_name].layout().addWidget(checkbox, row, col)
            iter += 1


    def get_checked_states(self):
        return {key: checkbox.isChecked() for key, checkbox in self.check_box_dict.items()}
    
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
    def __init__(self, widget, dataset_handler, include = None, exclude = None, title = "", max_rows = max_rows_checkboxes, title_filter = []):
        global title_filter_global
        super().__init__(widget)

        self.title = title
        self.dataset_handler = dataset_handler
        self.include = include
        self.exclude = exclude
        self.title_filter = title_filter + title_filter_global
        self.max_rows = max_rows

        self.setWidgetResizable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Hacerlo redimensionable solo horizontalmente

        # Crear un widget que contendrá los grupos de checkboxes
        self.scroll_widget = QWidget()
        self.scroll_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_widget.setMinimumHeight(max_rows) # por lo que sea poner un mínimo hace que evite el slider vertical lateral...
        self.setWidget(self.scroll_widget)
        self.scroll_layout = QGridLayout(self.scroll_widget)

        self.update_checkboxes()

    def update_checkboxes(self):
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            
        self.check_box_dict = {}

        iter = 0
        LabelGroup = QGroupBox(self.title)
        LabelGroup.setLayout(QGridLayout())
        LabelGroup.setStyleSheet("font-weight: bold;")
        self.scroll_layout.addWidget(LabelGroup)
        
        for key, dataset_info in self.dataset_handler.getInfo().items():
            group_name = dataset_info['model']
            if not group_name in self.check_box_dict:
                if (self.include and self.include not in group_name) or \
                (self.exclude and self.exclude in group_name): 
                    continue
                    
                title = group_name
                for filter in self.title_filter:
                    title = title.replace(filter, "")
                checkbox = QCheckBox(title)
                checkbox.setToolTip('/'.join(dataset_info['path'].split('/')[:-2])) # remove /test_name_blabla/results.yaml from path to show group path
                checkbox.setStyleSheet("font-weight: normal;") # Undo the bold text from parent 
                self.check_box_dict[group_name] = checkbox
                row = iter % self.max_rows
                col = iter // self.max_rows
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

