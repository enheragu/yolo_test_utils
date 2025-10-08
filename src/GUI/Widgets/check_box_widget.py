#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates a widget with slide view that contains all checkboxes with dataset parsed
"""

from copy import copy, deepcopy
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QGridLayout, QWidget, QCheckBox, QGroupBox, QScrollArea, QSizePolicy, QDialog

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

        if not self.check_box_dict:
            self.hide()
            
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

        if not self.check_box_dict:
            self.hide()

    def update_checkboxes(self):
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            
        self.check_box_dict = {}
        
        mainLabelGroup = QGroupBox(self.title)
        mainLabelGroup.setLayout(QHBoxLayout())
        mainLabelGroup.setStyleSheet("font-weight: bold;")
        self.scroll_layout.addWidget(mainLabelGroup)

        # orphanMainLabelGroup = QGroupBox('Extra')
        # orphanMainLabelGroup.setLayout(QGridLayout())
        # mainLabelGroup.layout().addWidget(orphanMainLabelGroup)

        groups = {'': {'parent': None, 'group': mainLabelGroup, 'extra': None, 'iter': 0}}

        visible_groups = set()
        for key, dataset_info in self.dataset_handler.getInfo().items():
            group_name = '/'.join(dataset_info['group_path'])
            check_exclusions = f"{group_name}/{dataset_info['model']}"
            if ((self.include and self.include not in check_exclusions) or
                (self.exclude and self.exclude in check_exclusions)):
                # print(f"Skipping {group_name = } as it does not match include ({self.include}) or exclude ({self.exclude}) tags")
                continue
            visible_groups.add(group_name)

        print(f"{visible_groups = }")

        for group_name in visible_groups:
            parts = group_name.split('/')
            current_path = ''
            parent = mainLabelGroup
            for part in parts[:-1]:
                current_path += '/' + part if current_path else part
                if current_path not in groups:
                    label_group = QGroupBox(part)
                    label_group.setLayout(QHBoxLayout())
                    label_group.setStyleSheet("font-weight: bold;")
                    parent.layout().addWidget(label_group)
                    groups[current_path] = {'parent': parent, 'group': label_group, 'extra': None, 'iter': 0}
                else:
                    label_group = groups[current_path]['group']
                parent = label_group
            
            # In the last level adds QGridLayout
            checboxLabelGroup = QGroupBox(parts[-1])
            checboxLabelGroup.setLayout(QGridLayout())
            parent.layout().addWidget(checboxLabelGroup)
            groups[group_name] = {'parent': parent, 'group': checboxLabelGroup, 'extra': None, 'iter': 0}

        for key, dataset_info in self.dataset_handler.getInfo().items():
            group_name = '/'.join(dataset_info['group_path'])
            item_name = f"{group_name}/{dataset_info['model']}"
            if group_name not in visible_groups:
                # print(f"Skipping {group_name = } as its not in visible groups set ({visible_groups})")
                continue
            if not item_name in self.check_box_dict:
                # if (self.include and self.include not in group_name) or \
                # (self.exclude and self.exclude in group_name): 
                #     print(f"Skipping {group_name = } as it does not match include ({self.include}) or exclude ({self.exclude}) tags")
                #     continue

                title = dataset_info['model']
                for f in self.title_filter:
                    title = title.replace(f, "")
                checkbox = QCheckBox(title)
                checkbox.setToolTip('/'.join(dataset_info['path'].split('/')[:-2]))
                checkbox.setStyleSheet("font-weight: normal;")
                self.check_box_dict[item_name] = checkbox
                row = groups[group_name]['iter'] % self.max_rows
                col = groups[group_name]['iter'] // self.max_rows
                if isinstance(groups[group_name]['group'].layout(), QGridLayout):
                    groups[group_name]['group'].layout().addWidget(checkbox, row, col)
                else:
                    groups[group_name]['extra'].layout().addWidget(checkbox, row, col)
                groups[group_name]['iter'] += 1
        
        print(f"{groups = }")
        print(f"{self.check_box_dict = }")
        # Clear empty layouts
        for group_name, group_info in groups.items():
            group_box = group_info['group']
            layout = group_box.layout()
            if layout and layout.count() == 0:
                parent = group_info['parent']
                if parent and parent.layout():
                    parent.layout().removeWidget(group_box)
                    group_box.deleteLater()
        
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

class BestGroupCheckBoxWidget(GroupCheckBoxWidget):
    """
        Makes use fo mAP50 by default to filter for best trial
    """
    def __init__(self, widget, dataset_handler, include = None, exclude = None, title = "", max_rows = max_rows_checkboxes, title_filter = [], class_selector = None):
        super().__init__(widget, dataset_handler, include, exclude, title, max_rows, title_filter)

        self.class_selector = class_selector

    def getChecked(self):
        best_item_group = {}
        best_metric_group = {}
        for group, checkbox in self.check_box_dict.items():
            if checkbox.isChecked():
                keys = [key for key in self.dataset_handler.keys() if group in key]
                for key in keys:
                    data = self.dataset_handler[key]
                    if not 'validation_best' in data:
                        continue
                    # print(data.keys())
                    if self.class_selector:
                        plot_class = self.class_selector.currentText() if  self.class_selector.currentText() in data['validation_best']['data'] else 'all'
                    else:
                        plot_class = 'all'

                    map_value = data['validation_best']['data'][plot_class]['mAP50']
                         
                    if not group in best_item_group or map_value > best_metric_group[group]:
                        best_item_group[group] = key
                        best_metric_group[group] = map_value
            
            # if group in best_item_group:
            #     print(f"[BestGroupCheckBoxWidget::getChecked] Best items selected: {best_item_group[group]} with metric {best_metric_group[group]}")
        # print(f"[BestGroupCheckBoxWidget::getChecked] Final best items selected: {best_item_group}")
        return list(best_item_group.values())