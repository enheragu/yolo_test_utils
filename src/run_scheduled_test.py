#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import traceback
from datetime import datetime

sys.path.append('.')
import src # Imports __init__.py defined in paralel to this script

from config_utils import log, handleArguments, bcolors, yolo_outpu_log_path
from YoloExecution.train_yolo import TestTrainYolo
from YoloExecution.validation_yolo import TestValidateYolo
from update_datset import checkKaistDataset
from test_scheduler import TestQueue

class Logger(object):
    def __init__(self):
        print("[Logger::__init__] New logger requested")
        timetag = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        log_file_name = f"{yolo_outpu_log_path}/{timetag}_test_yolo.log"

        self.terminal = sys.stdout
        self.log_file_name = log_file_name.replace("//", "/")
        
        self.executing_symlink = f'{yolo_outpu_log_path}/../now_executing.log'
        self.latest_symlink = f'{yolo_outpu_log_path}/../latest.log'

        if os.path.exists(self.executing_symlink):
            print("Test already being executed. Log to different file")
            self.executing_symlink = f'{yolo_outpu_log_path}/../now_executing_1.log'
            self.latest_symlink = f'{yolo_outpu_log_path}/../latest_1.log'
            
        os.symlink(self.log_file_name, self.executing_symlink)
    
    def retagOutputFile(self, new_tag = ""):
        # Remove previous executing link to make the new one
        if os.path.exists(self.executing_symlink):
            os.unlink(self.executing_symlink)
        new_name = self.log_file_name.replace(f'{yolo_outpu_log_path}/', f"{yolo_outpu_log_path}/{new_tag}_")
        os.rename(self.log_file_name, new_name)
        self.log_file_name = new_name
        os.symlink(self.log_file_name, self.executing_symlink)

    def __del__(self):
        if os.path.exists(self.executing_symlink):
            os.unlink(self.executing_symlink)
        if os.path.exists(self.latest_symlink):
            os.unlink(self.latest_symlink)

        os.symlink(self.log_file_name, self.latest_symlink)
    
    def write(self, message):
        self.terminal.write(message)
        
        with open(self.log_file_name, "a") as log:
            log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    


def log_nfty(msg, success = True):
    if 'NTFY_TOPIC' in os.environ:
        finish_ok = False
        import requests
        topic = os.getenv('NTFY_TOPIC')

        title = "Training execution " + ("finished" if success else "failed")
        priority = "default" if success else "high"
        tags = "+1,partying_face" if success else "-1,man_facepalming"
        requests.post(f"https://ntfy.sh/{topic}", data =str(msg).encode(encoding='utf-8'),
                    headers={"Title": title, "Priority": priority, "Tags": tags
                    })

def logCoolMessage(msg, bcolors = bcolors.OKCYAN):
    min_len = len(msg) + 6
    log(f"\n\n{'#'*min_len}\n#{' '*(min_len-2)}#\n#  {msg}  #\n#{' '*(min_len-2)}#\n{'#'*min_len}\n\n", bcolors)


if __name__ == '__main__':
    test_queue = TestQueue()

    if len(sys.argv) > 1:
        log(f"Add new test to queue: {sys.argv[1:]}")
        test_queue.add_new_test(sys.argv[1:])  

    nex_test = test_queue.get_next_test()
    while nex_test:
        sys.stdout = Logger()
        logCoolMessage(f"START TEST EXECUTION")
        try:    
            condition_list, option_list, model_list, opts = handleArguments(nex_test)
            checkKaistDataset(option_list, opts.dformat)

            for index in range(opts.iterations):
                for mode in opts.run_mode:
                    if mode == 'val':
                        TestValidateYolo(condition_list, option_list, model_list, opts.device, opts.cache, opts.pretrained, opts.path_name, opts.dformat)
                        
                    elif mode == 'train':
                        TestTrainYolo(condition_list, option_list, model_list, opts.device, opts.cache, opts.pretrained, opts.path_name, opts.dformat)
                        

                log(f"Options executed (iteration: {index}/{opts.iterations}) were:\n\t· {condition_list = }\n\t· {option_list = }\n\t· {model_list = };\n\t· run mode: {opts.run_mode}")
            logCoolMessage(f"CLEAN FINISH TEST EXECUTION")

            raw_msg = f"Options executed (n iterations: {opts.iterations}) were: {condition_list = }; {option_list = }; {model_list = }; run mode = {opts.run_mode}"
            log_nfty(raw_msg, success=True)

        except Exception as e:
            log(f"Options failed (at index {index}) were:\n\t· {condition_list = }\n\t· {option_list = }\n\t· {model_list = };\n\t· run mode: {opts.run_mode}", bcolors.ERROR)
            log(f"Catched exception: {e}", bcolors.ERROR)
            log(traceback.format_exc(), bcolors.ERROR)
            
            logCoolMessage(f"EXCEPTION. FAILED TEST EXECUTION", bcolors.ERROR)

            raw_msg = f"Options failed (at index {index}/{opts.iterations})  were: {condition_list = }; {option_list = }; {model_list = }; run mode = {opts.run_mode}\n"
            raw_msg += f"Catched exception: {e}"
            log_nfty(raw_msg, success=False)

            ### ALREADY FINISHED EVERYTHING, MOVE LOG FILE WITH ERROR TAG
            #   NEEDS TO BE CLOSED BEFOREHAND
            sys.stdout.retagOutputFile("exception")
        
        # Gets next test for next iteration
        nex_test = test_queue.get_next_test()