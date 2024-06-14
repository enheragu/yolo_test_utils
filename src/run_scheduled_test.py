#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import traceback

from Dataset.update_datset import checkKaistDataset
from test_scheduler import TestQueue, stop_env_var
from test_scheduler import isTimetableActive, sleep_until

from utils import Logger, log, log_ntfy, logCoolMessage, bcolors
from Dataset import generateCFGFiles, clearCFGFIles
from utils import getGPUTestIDTag
from argument_parser import handleArguments, yolo_outpu_log_path

sys.path.append('.')
import src # Imports __init__.py defined in paralel to this script

if __name__ == '__main__':
    logger = Logger(yolo_outpu_log_path)
    test_queue = TestQueue()

    if len(sys.argv) > 1:
        log(f"Add new test to queue: {sys.argv[1:]}")
        nex_test = sys.argv[1:]
    else:
        nex_test = test_queue.get_next_test()
    while nex_test:

        logCoolMessage(f"START TEST EXECUTION")
        index = 0
        try:
            condition_list, option_list, model_list, opts = handleArguments(nex_test)
            checkKaistDataset(option_list, opts.dformat, opts.thermal_eq, opts.rgb_eq)
            
            dataset_config_list = generateCFGFiles(condition_list, option_list, dataset_tag = opts.dformat)
        except Exception as e:
            log(f"Problem generating dataset or configuration files for {nex_test}.", bcolors.ERROR)
            log(f"Catched exception: {e}", bcolors.ERROR)
            log(traceback.format_exc(), bcolors.ERROR)
            
            logCoolMessage(f"EXCEPTION. FAILED TEST EXECUTION", bcolors.ERROR)

            raw_msg = f"Problem generating dataset or configuration files for {nex_test}\n"
            raw_msg += f"Catched exception: {e}"
            log_ntfy(raw_msg, success=False)
            test_queue.finished_test(False)

            ### ALREADY FINISHED EVERYTHING, MOVE LOG FILE WITH ERROR TAG
            #   NEEDS TO BE CLOSED BEFOREHAND
            sys.stdout.retagOutputFile("exception")
            dataset_config_list = [] # Set to empty to avoid loop

        try:    
            for dataset in dataset_config_list:
                for index in range(opts.iterations):
                    log("--------------------------------------------------------------------------")
                    log(f"Start iteration {index+1}/{opts.iterations}")
                    ret, init_time = isTimetableActive()
                    if not ret:
                        log_ntfy(title="Pause tests", msg=f"Pause requested for tests in {getGPUTestIDTag()}.", tags = "")
                        sleep_until(init_time)
                        log_ntfy(title="Resume tests", msg=f"Pause finished for tests in {getGPUTestIDTag()}.", tags = "")

                    for mode in opts.run_mode:
                        # Both YoloExecution functions are added here so that variables
                        # that need to be ovewrytten take place before adding all that
                        # code
                        if mode == 'val':
                            from YoloExecution.validation_yolo import TestValidateYolo
                            TestValidateYolo(dataset, model_list, opts)
                            
                        elif mode == 'train':
                            from YoloExecution.train_yolo import TestTrainYolo
                            TestTrainYolo(dataset, model_list, opts)
                    
                    # If stop is requested, pending iterations are added to queue, then
                    # queu handler will handle the stop not providing next test in queu e
                    if os.path.exists(stop_env_var):
                        missing_iterations = opts.iterations - index + 1

                        try:
                            index_iterations_option = next.nex_test('--iterations')
                            if index_iterations_option < len(nex_test) - 1:
                                nex_test[index_iterations_option + 1] = f"{index_iterations_option}"
                        except ValueError:
                            pass
                        test_queue.add_new_test(nex_test)
                        log("Env {stop_env_var} detected. Stopping execution.", bcolors.WARNING)
                        log_ntfy(title="Stop requested", msg=f"Stop requested for tests in {getGPUTestIDTag()}.", tags = "")
                        break

                    log(f"Options executed (iteration: {index+1}/{opts.iterations}) in {getGPUTestIDTag()} were:\n\t· {dataset = }\n\t· {model_list = };\n\t· run mode: {opts.run_mode}")
                    raw_msg = f"Options executed (iteration: {index+1}/{opts.iterations}) were: {dataset = }; {model_list = }; run mode = {opts.run_mode}"
                    log_ntfy(raw_msg, success=True)
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
        nex_test = test_queue.get_next_test()
        if nex_test:
            logger.renew()
    
    log_ntfy(title="Finished all tests", msg=f"No more test to execute in queue in {getGPUTestIDTag()}. Process will be finished, add more test to be executed if you have any pending 😀", tags = "tada,woman_dancing")