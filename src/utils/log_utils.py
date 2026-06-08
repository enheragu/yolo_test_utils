#!/usr/bin/env python3
# encoding: utf-8

import sys
import os
import re

from tqdm import tqdm
import tabulate

from datetime import datetime
from .id_tag import getGPUTestID

import logging

# """
#     Needs to close the file each time so that it not collides with others
# """
# class CloseAfterWriteFileHandler(logging.FileHandler):
#     def __init__(self, filename, mode='a', encoding=None, delay=False):
#         super().__init__(filename, mode, encoding, delay)

#     def emit(self, record):
#         super().emit(record)
#         self.stream.close()  # Cerrar el archivo después de cada escritura

# """
#     Needs to add file handler to YOLO logger
# """
# def add_file_handler(logger, filename, level=logging.INFO):
#     """Agrega un FileHandler al logger con el nombre de archivo especificado."""
#     file_handler = CloseAfterWriteFileHandler(filename)
#     file_handler.setLevel(level)
#     formatter = logging.Formatter('%(message)s')
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)

class Logger(object):
    def __init__(self, output_path):
        print("[Logger::__init__] New logger requested")

        self.terminal = sys.stdout
        
        self.output_path = output_path
        self.log_file_name = ""
        id = getGPUTestID()
        self.executing_symlink = f'{self.output_path}/../now_executing{id}.log'
        self.latest_symlink = f'{self.output_path}/../latest{id}.log'

        self._create_log_file()

        sys.stdout = self
        sys.stderr = self
        
        # Local dep, adding it globa may cause other issues as it comes with
        # lots of stuff
        from ultralytics.yolo.utils import LOGGER
        LOGGER.addHandler(logging.StreamHandler(self))

    def fileno(self):
        # Needed function for some libraries
        return self.terminal.fileno()
    
    def _create_log_file(self):
        timetag = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.log_file_name = f"{self.output_path}/{timetag}_test_yolo.log"

        # Use islink instead of exists. If it appears in red, pointing to nonexistent
        # file, exist will return false
        if os.path.islink(self.executing_symlink):
            log(f"Refresh previous 'executing' simlink with new on pointing to current execution log: {self.log_file_name}.", bcolors.WARNING)
            os.unlink(self.executing_symlink)
        os.symlink(self.log_file_name, self.executing_symlink)

    def renew(self):
        self.finishTest()  # Finaliza la prueba actual
        self._create_log_file()  # Crea un nuevo archivo de registro

    def retagOutputFile(self, new_tag = ""):
        # Remove previous executing link to make the new one
        if os.path.islink(self.executing_symlink):
            os.unlink(self.executing_symlink)
        new_name = self.log_file_name.replace(f'{self.output_path}/', f"{self.output_path}/{new_tag}_")
        os.rename(self.log_file_name, new_name)
        self.log_file_name = new_name
        os.symlink(self.log_file_name, self.executing_symlink)

    def finishTest(self):
        if os.path.islink(self.executing_symlink):
            os.unlink(self.executing_symlink)
        if os.path.islink(self.latest_symlink):
            os.unlink(self.latest_symlink)
            
        os.symlink(self.log_file_name, self.latest_symlink)
        
    def __del__(self):
        self.finishTest()
    
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file_name, "a", encoding="utf-8") as log_file:
            log_file.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass  



################################
#     Format Logging stuff     #
################################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def getTimetagNow():
    return datetime.now().strftime('%F %T.%f')[:-3]

def log(msg = "", color = bcolors.OKCYAN):
    timetag = getTimetagNow()
    tqdm.write(f"{color}[{timetag}] {msg}{bcolors.ENDC}")

    
def log_ntfy(msg, success = True, title = None, tags = None):
    if 'NTFY_TOPIC' in os.environ:
        finish_ok = False
        import requests
        topic = os.getenv('NTFY_TOPIC')

        set_title = "Training execution " + ("finished" if success else "failed") if not title else title
        priority = "default" if success else "high"
        if not tags:
            set_tags = ("+1,partying_face" if success else "-1,man_facepalming")
        else:
            set_tags = tags

        requests.post(f"https://ntfy.sh/{topic}", data =str(msg).encode(encoding='utf-8'),
                    headers={"Title": set_title, "Priority": priority, "Tags": set_tags
                    })

def logCoolMessage(msg, bcolors = bcolors.OKCYAN):
    min_len = len(msg) + 6
    log(f"\n\n{'#'*min_len}\n#{' '*(min_len-2)}#\n#  {msg}  #\n#{' '*(min_len-2)}#\n{'#'*min_len}\n\n", bcolors)


def printDictKeys(diccionario, nivel=0):
    if nivel==0:
        print(f"Print dict keys: ")
    for key, value in diccionario.items():
        print(f"{'    ' * (1+nivel)}· {key}")
        if isinstance(value, dict):
            printDictKeys(value, nivel + 1)


def logTable(row_data, output_path, filename, colalign = None, screen=True, showindex=False, formats=None):
    """
    formats: iterable of {'txt','tex','html'} to write. None = all three.
    """
    if not row_data:
        log(f"[logTable] No rows available for {filename}; skipping table export.", bcolors.WARNING)
        return

    fmts = set(formats) if formats is not None else {'txt', 'tex', 'html'}

    table_str = tabulate.tabulate(row_data, headers="firstrow", tablefmt="fancy_grid", colalign = colalign, showindex=showindex)
    if screen:
        log(f"\n{table_str}")

    file_name = os.path.join(output_path, filename.lower().replace(' ','_'))
    log(f"Stored data in {file_name} (formats: {sorted(fmts)})")

    if 'txt' in fmts:
        with open(f"/{file_name}.txt", 'w+') as file:
            file.write(f'{filename}\n')
            file.write(table_str)

    if 'tex' in fmts:
        table_latex = tabulate.tabulate(row_data, headers="firstrow", tablefmt="latex", colalign = colalign, showindex=showindex)
        headers = row_data[0]
        # Bold only the header row by wrapping each cell as a whole. A naive
        # str.replace bolds inner letters too (SUPERPIXEL → SU\textbf{P}E\textbf{R}\textbf{P}IXEL,
        # mAP50 → mA\textbf{P}50) because single-letter headers like 'P'/'R' match
        # everywhere. Splitting on '&' avoids substring substitution entirely.
        lines = table_latex.split('\n')
        hline_idxs = [i for i, ln in enumerate(lines) if r'\hline' in ln]
        if len(hline_idxs) >= 2:
            for i in range(hline_idxs[0] + 1, hline_idxs[1]):
                line = lines[i]
                if line.rstrip().endswith(r'\\'):
                    body, tail = line.rstrip()[:-2], r' \\'
                else:
                    body, tail = line, ''
                new_cells = []
                for cell in body.split('&'):
                    stripped = cell.strip()
                    if not stripped:
                        new_cells.append(cell)
                        continue
                    lead  = cell[:len(cell) - len(cell.lstrip())]
                    trail = cell[len(cell.rstrip()):]
                    new_cells.append(f"{lead}\\textbf{{{stripped}}}{trail}")
                lines[i] = '&'.join(new_cells) + tail
            table_latex = '\n'.join(lines)
        
        table_latex = re.sub(
            r'\\begin\{tabular\}\{[^}]*\}',
            r'\\begin{tabular}{|l|' + 'c' * (len(headers) - 1) + '|}',
            table_latex,
            count=1
        )
        caption = f"{filename}"
        label = f"tab:{filename.lower().replace(' ','_')}"
        table_latex_with_caption = f"\\begin{{table}}[ht]\n\\centering\n{table_latex}\n\\captionsetup{{justification=centering}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}"

        with open(f"{file_name}.tex", 'w') as file:
            file.write(table_latex_with_caption)

    if 'html' in fmts:
        table_to_html(row_data[1:], row_data[0], output_path, filename, showindex)

def table_to_html(table, headers, output_path, filename, showindex=False):

    file_name = os.path.join(output_path, filename.lower().replace(' ','_'))
    ansi_to_html = {
        "\033[92m": '<span class="green">',
        "\033[91m": '<span class="red">',
        "\033[9202m": '<span class="orange">',
        "\033[0m": '</span>'
    }

    def reemplazar_ansi(texto):
        for ansi, html in ansi_to_html.items():
            texto = texto.replace(ansi, html)
        return texto

    table_html = [
        [reemplazar_ansi(str(cell)) for cell in row]
        for row in table
    ]

    html_table = tabulate.tabulate(table_html, headers=headers, tablefmt="unsafehtml", showindex=showindex)
    # Find header row
    start = html_table.find("<tr>")
    end = html_table.find("</tr>", start) + len("</tr>")
    header_row = html_table[start:end]

    # Insert header row also at the end as a footer :)
    html_table_with_footer = html_table.replace("</table>", f"{header_row}\n</table>")

    css = """
    <style>
    .green { color: #28a745; font-weight: bold; }
    .red { color: #dc3545; font-weight: bold; }
    .orange { color: orange; font-weight: bold; }
    table { border-collapse: collapse; }
    th, td { border: 1px solid #333; padding: 8px; text-align: center; }
    th { background-color: #f2f2f2; }
    </style>
    """

    with open(f"{file_name}.html", "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html>\n<html>\n<head>{css}</head>\n<body>\n{html_table_with_footer}\n</body>\n</html>")
    
