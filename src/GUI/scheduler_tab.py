#!/usr/bin/env python3
# encoding: utf-8

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QSizePolicy, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QLineEdit, QListWidget, QCheckBox, QGroupBox
from PyQt6.QtGui import QColor, QAction

from config_utils import parseYaml, dumpYaml, configArgParser
from log_utils import log, bcolors
from GUI.Widgets.checable_combobox import CheckableComboBox

from test_scheduler import TestQueue

class SchedulerHandlerPlotter(QWidget):

    def __init__(self):
        super().__init__()
        
        self.parser = configArgParser()
        self.options = {}
        for action in self.parser._actions:
                if action.option_strings and len(action.option_strings) > 1:
                    long_option = str(action.option_strings[1]).replace("--","").title()
                    if long_option != "Help":
                        choices = action.choices  # Obtener las opciones 'choices' del objeto 'action'
                        # Convertir las opciones a una lista si es un conjunto o un diccionario
                        if isinstance(choices, (set, dict)):
                            choices = list(choices)
                        self.options[long_option] = {'title': long_option, 'long':action.option_strings[1], 'short':action.option_strings[0], 
                                                     'action': action, 'help': action.help, 'type': action.type, 'choices': choices,
                                                     'nargs': action.nargs, 'default': action.default}


        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # self.setWidgetResizable(True)  # Permitir que el widget se expanda
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # self.load_button = QPushButton("Cargar y Mostrar Datos")
        # self.load_button.clicked.connect(self.load_and_display_data)
        # self.layout.addWidget(self.load_button)

        self.argparseUI()

        self.title_label = QLabel("Pending test execution:")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.title_label)
        self.table_pending = QTableWidget()
        self.layout.addWidget(self.table_pending)

        self.title_label = QLabel("Currently executing:")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.title_label)
        self.table_executing = QTableWidget()
        self.layout.addWidget(self.table_executing)

        
    def argparseUI(self):
        num_rows = 2
        num_cols = len(self.options.values()) // num_rows
        
        self.title_label = QLabel("Add new test to pending list:")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.title_label)
        post_test_widget = QGroupBox()
        self.layout.addWidget(post_test_widget)

        post_test_layout = QGridLayout()
        post_test_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Alineación vertical superior
        post_test_widget.setLayout(post_test_layout)
        post_test_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        row = 0
        col = 0
        for index, option in enumerate(self.options.values()):
            input_widget = QWidget()
            if col > num_cols:
                row += 1
                col = 0
            post_test_layout.addWidget(input_widget, row, col)
            col+=1

            input_layout = QVBoxLayout()
            input_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Alineación vertical superior
            input_widget.setLayout(input_layout)
            input_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            title_label = QLabel(f"{option['title']}:")
            title_label.setStyleSheet("font-weight: bold;")
            input_layout.addWidget(title_label)

            # label = QLabel(option['help'])
            # label.setWordWrap(True)
            # input_layout.addWidget(label)

            if option['choices'] and option['nargs'] == '*':
                input_field = CheckableComboBox()
                input_field.addItems(option['choices'])
            elif option['choices']:
                input_field = QComboBox()
                input_field.addItems(option['choices'])
            elif option['type'] == str:
                input_field = QLineEdit()
            elif option['type'] == int:
                input_field = QLineEdit()
            elif option['type'] == bool:
                input_field = QComboBox()
                input_field.addItems(['True', 'False'])
            else:
                input_field = QWidget()
                print(f"Type no handled for: {option['title']} with type {option['type']}")

            input_layout.addWidget(input_field)
           
            if 'default' in option:
                default_value = option['default']
                ## Default is set to all, so is a bit inconvenient to have to uncheck all 
                #  of them when is a CheckableComboBox
                # if isinstance(input_field, CheckableComboBox):
                #     input_field.setDefaults(default_value)
                if isinstance(input_field, QComboBox):
                    input_field.setCurrentText(str(default_value))
                elif isinstance(input_field, QLineEdit):
                    input_field.setText(str(default_value))
                elif isinstance(input_field, QListWidget):
                    for value in default_value:
                        items = input_field.findItems(str(value), Qt.MatchExactly)
                        for item in items:
                            item.setSelected(True)

            option['widget'] = input_field

        submit_button = QPushButton('Submit')
        submit_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        post_test_layout.addWidget(submit_button, 0, num_cols +1, num_rows, 1)
        submit_button.clicked.connect(self.submit_new_test)

    def submit_new_test(self):
        cmd_list = []
        for option in self.options.values():

            value = None
            if isinstance(option['widget'], CheckableComboBox):
                value = option['widget'].currentData()
                if len(value) == 1:
                    value = str(value[0])
            elif isinstance(option['widget'], QComboBox):
                value = option['widget'].currentText()
            elif isinstance(option['widget'], QLineEdit):
                value = option['widget'].text()
            elif isinstance(option['widget'], QCheckBox):
                value = option['widget'].isChecked()
            else:
                pass

            # Just ignore the nones... let argparse handle them later
            if value is not None and value != 'None':
                cmd_list.append(option['long'])
                cmd_list.append(value)

        # print(cmd_list)
        test_queue = TestQueue()
        test_queue.add_new_test(cmd_list) 
        self.load_and_display_data()

    def load_and_display_data(self):
        # Limpiar la tabla antes de cargar nuevos datos
        def clearTable(table):
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)

        clearTable(self.table_pending)
        clearTable(self.table_executing)

        home_dir = os.getenv('HOME')
        # Definir los nombres de los archivos
        file_pending = f"{home_dir}/.cache/eeha_yolo_test/pending.yaml"
        file_executing = f"{home_dir}/.cache/eeha_yolo_test/executing.yaml"

        def _fill_table(data, table):
            
            
            num_rows = len(data)
            num_cols = len(self.options.keys())

            table.setRowCount(num_rows)
            table.setColumnCount(num_cols)

            table.setHorizontalHeaderLabels(self.options.keys())

            title_index_map = {title: i for i, title in enumerate(self.options)}
            for row, test in enumerate(data):
                for i, (command, content) in enumerate(zip(test[::2], test[1::2])):
                    for column_title, column_info in self.options.items():
                        if command == column_info['long'] or command == column_info['short']:
                            column_index = title_index_map[column_title]
                            item = QTableWidgetItem(str(content))
                            cell_color = QColor(144, 238, 144)
                            item.setBackground(cell_color)
                            table.setItem(row, column_index, item)
                            break

            # Fill empty cells with default
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    item = table.item(row, column)
                    if item is None or item.text() == "":
                        column_title = table.horizontalHeaderItem(column).text()
                        default_value = self.options[column_title]['action'].default
                        if default_value is not None:
                            item = QTableWidgetItem(str(default_value))
                            table.setItem(row, column, item)

            table.resizeColumnsToContents()
            table.resizeRowsToContents()
            

        if os.path.exists(file_pending):
            _fill_table(parseYaml(file_pending), self.table_pending)

        if os.path.exists(file_executing):
            _fill_table(parseYaml(file_executing), self.table_executing)
    
    def update_view_menu(self, archive_menu, view_menu, tools_menu):
        self.load_and_display_data()

        self.update_view_action = QAction("Update view", self)
        self.update_view_action.setShortcut(Qt.Key.Key_F5)
        self.update_view_action.triggered.connect(self.load_and_display_data)
        tools_menu.addAction(self.update_view_action)