#!/usr/bin/env python3
# encoding: utf-8

"""
    Prints in terminal the different files (pending, executing...) in tables
"""
import os
import glob
from tabulate import tabulate

from utils import log, bcolors, logTable
from GUI.scheduler_tab import parseTestFile, parseDataMatrix
from utils.yaml_utils import parseYaml
from test_scheduler import pending_file_default, pending_stopped_default, executing_file_default, finished_file_ok_default, finished_file_failed_default


def filterOutputMatrix(matrix, black_list=['Pretrained', 'Test-Name', 'Dataset', 'Deterministic', 'Run-Mode']):
    header = matrix[0]
    indexes = [i for i, title in enumerate(header) if title not in black_list]
    filtered_mat = [[row[i] for i in indexes] for row in matrix]
    return filtered_mat

def printColoredTable(file, title_in):
    if not os.path.exists(file):
        log(f"File not found, table wont be displayed: {file}", bcolors.ERROR)
        return

    print("\n")
    log(f"Get test queue from {file}")
    
    title = f"  {title_in}  "
    data = parseYaml(file)
    if isinstance(data, list):
        matrix, nondefault = parseDataMatrix(data, file)
        # matrix, nondefault = parseTestFile(file)
    elif isinstance(data, dict):
        matrix, nondefault = parseDataMatrix([data['test']], file)
    else:
        log(f"Data is not of a handled type: {type(data)}", bcolors.ERROR)
        
    if not matrix:
        log(f"Empty matrix, no table to display from {file}.", bcolors.ERROR)
        return

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if nondefault[i][j] == 1:
                matrix[i][j] = f"{bcolors.OKGREEN}{matrix[i][j]}{bcolors.ENDC}"

    matrix = filterOutputMatrix(matrix)
    formatted_table = tabulate(matrix, headers="firstrow", colalign=("center",), numalign="center", tablefmt='fancy_grid', showindex=True)

    table_width = len(formatted_table.splitlines()[1]) # Get length from dashes, which is second one
    title_dashes = '-' * ((table_width - len(title)) // 2)

    print(f"{bcolors.OKCYAN}{title_dashes}{title}{title_dashes}{bcolors.ENDC}\n")
    for line in formatted_table.splitlines():
        print(line)
    print("\n\n")

    cache_path = f"{os.getenv('HOME')}/.cache/eeha_yolo_test"
    logTable(row_data=matrix, output_path=cache_path, filename=f"{title_in.lower().replace(' ','_')}", screen=False, showindex=True)

if __name__ == "__main__":
    printColoredTable(pending_file_default, "Pending Test table")

    def iterate_executing(patron):
        # Iterar sobre los archivos que coinciden con el patrón
        for archivo in glob.glob(patron):
            yield archivo

    executing_file_template = executing_file_default.replace(".yaml", "*.yaml")
    for executing in iterate_executing(executing_file_template):
        printColoredTable(executing, "Executing Test table")
