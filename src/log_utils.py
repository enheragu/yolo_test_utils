#!/usr/bin/env python3
# encoding: utf-8

import sys
import os

from datetime import datetime
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
        self.executing_symlink = f'{self.output_path}/../now_executing.log'
        self.latest_symlink = f'{self.output_path}/../latest.log'

        self._create_log_file()

        sys.stdout = self
        sys.stderr = self
        
        # Local dep, adding it globa may cause other issues as it comes with
        # lots of stuff
        from ultralytics.yolo.utils import LOGGER
        LOGGER.addHandler(logging.StreamHandler(self))

    def _create_log_file(self):
        timetag = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.log_file_name = f"{self.output_path}/{timetag}_test_yolo.log"
        if os.path.exists(self.executing_symlink):
            log(f"Refresh previous 'executing' simlink with new on pointing to current execution log: {self.log_file_name}.", bcolors.WARNING)
            os.unlink(self.executing_symlink)
        os.symlink(self.log_file_name, self.executing_symlink)

    def renew(self):
        self.finishTest()  # Finaliza la prueba actual
        self._create_log_file()  # Crea un nuevo archivo de registro

    def retagOutputFile(self, new_tag = ""):
        # Remove previous executing link to make the new one
        if os.path.exists(self.executing_symlink):
            os.unlink(self.executing_symlink)
        new_name = self.log_file_name.replace(f'{self.output_path}/', f"{self.output_path}/{new_tag}_")
        os.rename(self.log_file_name, new_name)
        self.log_file_name = new_name
        os.symlink(self.log_file_name, self.executing_symlink)

    def finishTest(self):
        if os.path.exists(self.executing_symlink):
            os.unlink(self.executing_symlink)
        if os.path.exists(self.latest_symlink):
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


def log(msg = "", color = bcolors.OKCYAN):
    timetag = datetime.utcnow().strftime('%F %T.%f')[:-3]
    print(f"{color}[{timetag}] {msg}{bcolors.ENDC}")

    
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
