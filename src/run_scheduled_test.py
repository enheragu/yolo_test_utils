#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import traceback

from datetime import datetime

from Dataset.update_datset import checkDataset
from test_scheduler import TestQueue, stop_env_var
from test_scheduler import isTimetableActive, sleep_until

from utils import Logger, log, log_ntfy, logCoolMessage, bcolors, getGPUTestID
from Dataset import generateCFGFiles, clearCFGFIles
from utils import getGPUTestIDTag
from argument_parser import handleArguments, yolo_outpu_log_path

sys.path.append('.')
import src # Imports __init__.py defined in paralel to this script

def ask_yes_no(question):
    while True:
        print(f"{question} (y/n): ")
        response = input().strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Answer with 'y' or 'n'.")



import threading
import psutil
import faulthandler
"""
    Review unfinished threads and processes status launched from this script.

    terminate_process (bool): If true forces termination of all child processes found
"""
def monitor_threads_and_processes(terminate_process=False):
    parent = psutil.Process()
    children = parent.children(recursive=True)
    
    log(f"Total child processes: {len(children)}", bcolors.WARNING)
    for child in children:
        log(f"PID: {child.pid}, Name: {child.name()}, Status: {child.status()}", bcolors.WARNING)
        log(f"\t· Cmdline: {child.cmdline()}", bcolors.WARNING)
        log(f"\t· CPU Time: {child.cpu_times()}", bcolors.WARNING)
        log(f"\t· Memory Info: {child.memory_info()}", bcolors.WARNING)
        
        if terminate_process:
            log(f"\t· Terminating process {child.pid}...", bcolors.WARNING)
            child.terminate()
            child.wait()  
            log(f"\t· Process {child.pid} terminated.\n", bcolors.WARNING)

    active_threads = threading.enumerate()
    log(f"Total active threads: {len(active_threads)}", bcolors.WARNING)
    for thread in active_threads:
        log(f"\t· Thread Name: {thread.name}, Thread ID: {thread.ident}, Is daemon: {thread.daemon}", bcolors.WARNING)
    
    # log("Thread Tracebacks:", bcolors.WARNING)
    # faulthandler.dump_traceback()

if __name__ == '__main__':
    logger = Logger(yolo_outpu_log_path)
    test_queue = TestQueue()

    if len(sys.argv) > 1:
        log(f"Add new test to queue: {sys.argv[1:]}")
        next_test = sys.argv[1:]
        resume_path = None
    else:
        next_test, resume_path = test_queue.check_resume_test()
        if next_test and not ask_yes_no('Do you want te resume test (y) or cancel and get next pending test (n)?'):
            next_test = None
            resume_path = None
        if next_test is None:
            next_test = test_queue.get_next_test()
    while next_test:

        logCoolMessage(f"START TEST EXECUTION")
        index = 0
        try:
            condition_list, option_list, model_list, opts = handleArguments(next_test)

            if not opts.dataset:
                checkDataset(option_list, opts.dformat, opts.thermal_eq, opts.rgb_eq,
                                  opts.distortion_correct, opts.relabeling)
                
                dataset_config_list = generateCFGFiles(condition_list, option_list, dataset_tag = opts.dformat)
            else:
                dataset_config_list = [opts.dataset]

        except Exception as e:
            log(f"Problem generating dataset or configuration files for {next_test}.", bcolors.ERROR)
            log(f"Catched exception: {e}", bcolors.ERROR)
            log(traceback.format_exc(), bcolors.ERROR)
            
            logCoolMessage(f"EXCEPTION. FAILED TEST EXECUTION", bcolors.ERROR)

            raw_msg = f"Problem generating dataset or configuration files for {next_test}\n"
            raw_msg += f"Catched exception: {e}"
            log_ntfy(raw_msg, success=False)
            test_queue.finished_test(False)

            ### ALREADY FINISHED EVERYTHING, MOVE LOG FILE WITH ERROR TAG
            #   NEEDS TO BE CLOSED BEFOREHAND
            sys.stdout.retagOutputFile("exception")
            dataset_config_list = [] # Set to empty to avoid loop

        try:    
            for dataset in dataset_config_list:
                for yolo_model in model_list:
                    for index in range(opts.iterations):
                        log("--------------------------------------------------------------------------")
                        log(f"Start iteration {index+1}/{opts.iterations}")
                        ret, init_time = isTimetableActive()
                        if not ret:
                            log_ntfy(title="Pause tests", msg=f"Pause requested for tests in {getGPUTestIDTag()}.", tags = "")
                            sleep_until(init_time)
                            log_ntfy(title="Resume tests", msg=f"Pause finished for tests in {getGPUTestIDTag()}.", tags = "")

                        id = getGPUTestID()
                        
                        if resume_path is None:
                            opts.resume = False
                            if opts.test_name is None:
                                test_name = dataset.split("/")[-1].replace(".yaml","").replace("dataset_","") + id
                            else:
                                test_name = opts.test_name + id

                            if opts.path_name is None:
                                if 'val' in opts.run_mode:
                                    path_name = "validate_" + yolo_model + "/" + test_name
                                elif 'train' in opts.run_mode:
                                    path_name = f'train_based_{yolo_model}/{test_name}'
                            else:
                                path_name = opts.path_name + "/" + test_name
                            path_name + "_" + datetime.now().strftime("%Y%m%d")
                        else:
                            opts.resume = False
                            path_name = resume_path 
                            resume_path = None # Reset :)
                            yolo_model = f'{yolo_outpu_log_path}/{path_name}/weights/last.pt'
                            log(f"Load previously executing test to continue in {path_name}")

                        test_queue.updateCurrentExecutingPath(path_name)
                        for mode in opts.run_mode:
                            # Both YoloExecution functions are added here so that variables
                            # that need to be ovewrytten take place before adding all that
                            # code
                            if mode == 'val':
                                from YoloExecution.validation_yolo import TestValidateYolo
                                TestValidateYolo(dataset, yolo_model, path_name, opts)
                                
                            elif mode == 'train':
                                from YoloExecution.train_yolo import TestTrainYolo
                                TestTrainYolo(dataset, yolo_model, path_name, opts)
                        
                        # If stop is requested, pending iterations are added to queue, then
                        # queu handler will handle the stop not providing next test in queu e
                        if os.path.exists(stop_env_var):
                            missing_iterations = opts.iterations - index + 1

                            try:
                                index_iterations_option = next.next_test('--iterations')
                                if index_iterations_option < len(next_test) - 1:
                                    next_test[index_iterations_option + 1] = f"{index_iterations_option}"
                            except ValueError:
                                pass
                            test_queue.add_new_test(next_test)
                            log("Env {stop_env_var} detected. Stopping execution.", bcolors.WARNING)
                            log_ntfy(title="Stop requested", msg=f"Stop requested for tests in {getGPUTestIDTag()}.", tags = "")
                            break

                        log(f"Options executed (iteration: {index+1}/{opts.iterations}) in {getGPUTestIDTag()} were:\n\t· {dataset = }\n\t· {model_list = };\n\t· run mode: {opts.run_mode}")
                        raw_msg = f"Options executed (iteration: {index+1}/{opts.iterations}) were: {dataset = }; {model_list = }; run mode = {opts.run_mode}"
                        log_ntfy(raw_msg, success=True)
            monitor_threads_and_processes(terminate_process=True)
            logCoolMessage(f"CLEAN FINISH TEST EXECUTION")
            clearCFGFIles(dataset_config_list)
            test_queue.finished_test(True)
            # raw_msg = f"Options executed (n iterations: {opts.iterations}) were: {dataset = }; {model_list = }; run mode = {opts.run_mode}"
            # log_ntfy(raw_msg, success=True)

        except Exception as e:
            log(f"Options failed (at index {index}) were:\n\t· {dataset = }\n\t· {model_list = };\n\t· run mode: {opts.run_mode}", bcolors.ERROR)
            log(f"Catched exception: {e}", bcolors.ERROR)
            log(traceback.format_exc(), bcolors.ERROR)
            
            logCoolMessage(f"EXCEPTION. FAILED TEST EXECUTION", bcolors.ERROR)

            raw_msg = f"Options failed (at index {index}/{opts.iterations}) in {getGPUTestIDTag()}  were: {dataset = }; {model_list = }; run mode = {opts.run_mode}\n"
            raw_msg += f"Catched exception: {e}"
            log_ntfy(raw_msg, success=False)
            test_queue.finished_test(False)

            ### ALREADY FINISHED EVERYTHING, MOVE LOG FILE WITH ERROR TAG
            #   NEEDS TO BE CLOSED BEFOREHAND
            sys.stdout.retagOutputFile("exception")
        
        # Gets next test for next iteration
        ## TBD check memory leak when continuous execution
        next_test = False #test_queue.get_next_test()
        if next_test:
            logger.renew()
    
    #log_ntfy(title="Finished all tests", msg=f"No more test to execute in queue in {getGPUTestIDTag()}. Process will be finished, add more test to be executed if you have any pending 😀", tags = "tada,woman_dancing")