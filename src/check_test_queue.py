#!/usr/bin/env python3
# encoding: utf-8

"""
    Prints in terminal the different files (pending, executing...) in tables
"""
import os
from tabulate import tabulate

from log_utils import log, bcolors
from GUI.scheduler_tab import parseTestFile
from test_scheduler import pending_file_default, pending_stopped_default, executing_file_default, finished_file_ok_default, finished_file_failed_default

def printColoredTable(file, title):

    if not os.path.exists(file):
        log(f"File not found, table wont be displayed: {file}", bcolors.ERROR)
        return

    title = f"  {title}  "
    matrix, nondefault = parseTestFile(file)

    if not matrix:
        log(f"Empty matrix, no table to display.", bcolors.ERROR)
        return

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if nondefault[i][j] == 1:
                matrix[i][j] = f"{bcolors.OKGREEN}{matrix[i][j]}{bcolors.ENDC}"


    formatted_table = tabulate(matrix, headers="firstrow", colalign=("center",), numalign="center")

    table_width = len(formatted_table.splitlines()[1]) # Get length from dashes, which is second one
    title_dashes = '-' * ((table_width - len(title)) // 2)



    print("\n")
    print(f"{bcolors.OKCYAN}{title_dashes}{title}{title_dashes}{bcolors.ENDC}\n")
    for line in formatted_table.splitlines():
        print(line)
    print("\n\n")

if __name__ == "__main__":
    printColoredTable(pending_file_default, "Pending Test table")

    printColoredTable(executing_file_default, "Executing Test table")
