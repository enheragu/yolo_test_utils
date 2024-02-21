#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import traceback

from update_datset import checkKaistDataset
from test_scheduler import TestQueue

from log_utils import Logger, log, log_ntfy, logCoolMessage, bcolors
from config_utils import handleArguments, yolo_outpu_log_path

sys.path.append('.')
import src # Imports __init__.py defined in paralel to this script

if __name__ == '__main__':
    logger = Logger(yolo_outpu_log_path)
    test_queue = TestQueue()

    if len(sys.argv) > 1:
        log(f"Add new test to queue: {sys.argv[1:]}")
        test_queue.add_new_test(sys.argv[1:])  

    nex_test = test_queue.get_next_test()
    while nex_test:

        logCoolMessage(f"START TEST EXECUTION")
        index = 0
        try:    
            condition_list, option_list, model_list, opts = handleArguments(nex_test)
            checkKaistDataset(option_list, opts.dformat)

            for index in range(opts.iterations):

                for mode in opts.run_mode:
                    # Both YoloExecution functions are added here so that variables
                    # that need to be ovewrytten take place before adding all that
                    # code
                    if mode == 'val':
                        from YoloExecution.validation_yolo import TestValidateYolo
                        TestValidateYolo(condition_list, option_list, model_list, opts.device, opts.cache, opts.pretrained, opts.path_name, opts.dformat)
                        
                    elif mode == 'train':
                        from YoloExecution.train_yolo import TestTrainYolo
                        TestTrainYolo(condition_list, option_list, model_list, opts.device, opts.cache, opts.pretrained, opts.path_name, opts.dformat, opts.batch)
                
                # If stop is requested, pending iterations are added to queue, then
                # queu handler will handle the stop not providing next test in queu e
                if os.getenv("EEHA_TEST_STOP_REQUESTED"):
                    missing_iterations = opts.iterations - index + 1

                    try:
                        index_iterations_option = next.nex_test('--iterations')
                        if index_iterations_option < len(nex_test) - 1:
                            nex_test[index_iterations_option + 1] = f"{index_iterations_option}"
                    except ValueError:
                        pass
                    test_queue.add_new_test(nex_test)
                    log("Env EEHA_TEST_STOP_REQUESTED detected. Stopping execution.", bcolors.WARNING)
                    break

                log(f"Options executed (iteration: {index+1}/{opts.iterations}) were:\n\t· {condition_list = }\n\t· {option_list = }\n\t· {model_list = };\n\t· run mode: {opts.run_mode}")
                raw_msg = f"Options executed (iteration: {index+1}/{opts.iterations}) were: {condition_list = }; {option_list = }; {model_list = }; run mode = {opts.run_mode}"
                log_ntfy(raw_msg, success=True)
            logCoolMessage(f"CLEAN FINISH TEST EXECUTION")

            test_queue.finished_test(True)
            # raw_msg = f"Options executed (n iterations: {opts.iterations}) were: {condition_list = }; {option_list = }; {model_list = }; run mode = {opts.run_mode}"
            # log_ntfy(raw_msg, success=True)

        except Exception as e:
            log(f"Options failed (at index {index}) were:\n\t· {condition_list = }\n\t· {option_list = }\n\t· {model_list = };\n\t· run mode: {opts.run_mode}", bcolors.ERROR)
            log(f"Catched exception: {e}", bcolors.ERROR)
            log(traceback.format_exc(), bcolors.ERROR)
            
            logCoolMessage(f"EXCEPTION. FAILED TEST EXECUTION", bcolors.ERROR)

            raw_msg = f"Options failed (at index {index}/{opts.iterations})  were: {condition_list = }; {option_list = }; {model_list = }; run mode = {opts.run_mode}\n"
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
    
    log_ntfy(title="Finished all tests", msg="No more test to execute in queue. Process will be finished, add more test to be executed if you have any pending 😀", tags = "tada,woman_dancing")