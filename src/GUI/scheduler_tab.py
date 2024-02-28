#!/usr/bin/env python3
# encoding: utf-8

import os
import uuid

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox, QSizePolicy, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QLineEdit, QListWidget, QCheckBox, QGroupBox, QStyle, QApplication
from PyQt6.QtGui import QColor, QAction, QIcon

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
        self.options_widget = QGroupBox()
        self.layout.addWidget(self.options_widget)

        post_test_layout = QGridLayout()
        post_test_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Alineación vertical superior
        self.options_widget.setLayout(post_test_layout)
        self.options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
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

        def _fill_table(data, table, editable):            
            num_rows = len(data)
            num_cols = len(self.options.keys())+1
            column_tiles = list(self.options.keys()) + ['ROW_ID']
            id_hidden_col = num_cols-1
            if editable:
                num_cols += 1 # Plus button column buttons
                column_tiles.append(' ')
                id_hidden_col = num_cols-2


            table.setRowCount(num_rows)
            table.setColumnCount(num_cols)

            table.setHorizontalHeaderLabels(column_tiles)

            for row, test in enumerate(data):

                for i, (command, content) in enumerate(zip(test[::2], test[1::2])):
                    for column_title, column_info in self.options.items():
                        if command == column_info['long'] or command == column_info['short']:
                            column_index = list(self.options.keys()).index(column_title)
                            item = QTableWidgetItem(str(content))
                            cell_color = QColor(144, 238, 144)
                            item.setBackground(cell_color)
                            table.setItem(row, column_index, item)
                            break
                
                new_row_id = str(uuid.uuid4())
                new_id_item = QTableWidgetItem(new_row_id)
                table.setItem(row, id_hidden_col, new_id_item)

                if editable:
                    self.cell_buttons(table, row, num_cols = num_cols, row_id = new_row_id, id_hidden_col = id_hidden_col)

            # Fill empty cells with default
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    item = table.item(row, column)
                    if item is None or item.text() == "":
                        column_title = table.horizontalHeaderItem(column).text()
                        if column_title in self.options:
                            default_value = self.options[column_title]['action'].default
                            if default_value is not None:
                                item = QTableWidgetItem(str(default_value))
                                table.setItem(row, column, item)

            table.resizeColumnsToContents()
            table.resizeRowsToContents()
            
            table.setColumnHidden(id_hidden_col, True)

        if os.path.exists(file_pending):
            _fill_table(parseYaml(file_pending), self.table_pending, editable = True)

        if os.path.exists(file_executing):
            _fill_table(parseYaml(file_executing), self.table_executing, editable = False)
    
    def cell_buttons(self, table, row, num_cols, row_id, id_hidden_col):
        edit_button = QPushButton()
        edit_button.setIcon(QIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)))
        edit_button.setToolTip('Update test changes')
        edit_button.clicked.connect(lambda _, r=row_id: self.update_test(r))

        delete_button = QPushButton()
        delete_button.setIcon(QIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton)))
        delete_button.setToolTip('Delete row')
        delete_button.clicked.connect(lambda _, r=row_id: self.delete_test(r))
        
        duplicate_button = QPushButton()
        duplicate_button.setIcon(QIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward)))
        duplicate_button.setToolTip('Duplicate row')
        duplicate_button.clicked.connect(lambda _, r=row_id, id=id_hidden_col: self.duplicate_row(r,id))

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0) # tight layout
        button_layout.addWidget(edit_button)
        button_layout.addWidget(duplicate_button)
        button_layout.addWidget(delete_button)
        button_container = QWidget()
        button_container.setLayout(button_layout)
        table.setCellWidget(row, num_cols-1, button_container)

    def duplicate_row(self, row_id, id_hidden_col):
        # log(f"[{self.__class__.__name__}::duplicate_row] Row with id: {row_id}")
        for row in range(self.table_pending.rowCount()):
            item = self.table_pending.item(row, len(self.options.keys()))  # Última columna es el ID
            if item is not None and item.text() == row_id:
                num_cols = self.table_pending.columnCount()

                # Get data from duplicable row
                row_data = []
                for col in range(num_cols):
                    item = self.table_pending.item(row, col)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append('')


                self.table_pending.insertRow(row + 1)
                for col, data in enumerate(row_data):
                    new_item = QTableWidgetItem(data)
                    self.table_pending.setItem(row + 1, col, new_item)
                    
                # Insert data again with new id
                new_row_id = str(uuid.uuid4())
                new_id_item = QTableWidgetItem(new_row_id)
                self.table_pending.setItem(row + 1, id_hidden_col, new_id_item)

                self.cell_buttons(self.table_pending, row + 1, num_cols, new_row_id)
                return

    def update_test(self, row):
        log(f"[{self.__class__.__name__}::edit_test] Row: {row}")

    def delete_test(self, row_id):
        # log(f"[{self.__class__.__name__}::delete_test] Row with id: {row_id}")

        for row in range(self.table_pending.rowCount()):
            item = self.table_pending.item(row, len(self.options.keys()))  # Última columna es el ID
            if item is not None and item.text() == row_id:
                # Eliminar la fila encontrada
                self.table_pending.removeRow(row)
                return

    def toggle_options(self):
        # Cambiar el estado del check basado en si las opciones están visibles o no
        if self.options_widget.isVisible():
            self.options_widget.hide()
        else:
            self.options_widget.show()

    def update_view_and_menu(self, archive_menu, view_menu, tools_menu):
        self.load_and_display_data()

        self.show_options_action = QAction('Show Options Tab', self, checkable=True)
        self.show_options_action.setShortcut(Qt.Key.Key_F11)
        self.show_options_action.setChecked(True) 
        self.show_options_action.triggered.connect(self.toggle_options)
        view_menu.addAction(self.show_options_action)

        self.update_view_action = QAction("Update view", self)
        self.update_view_action.setShortcut(Qt.Key.Key_F5)
        self.update_view_action.triggered.connect(self.load_and_display_data)
        tools_menu.addAction(self.update_view_action)