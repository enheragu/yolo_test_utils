#!/usr/bin/env python3
# encoding: utf-8

import os

# --- Cap BLAS/OMP threads per process (avoid nested oversubscription) ---
# The fusion runs Pool(cpu_count) worker processes. BLAS-heavy methods (e.g.
# fa: sklearn PCA/FactorAnalysis + np.linalg.eigh) would otherwise each spawn
# cpu_count BLAS threads -> cpu_count^2 threads (observed load ~131 on 16 HW
# threads = thrashing). One BLAS thread per worker keeps the process-level
# parallelism (Pool) without nested thread contention. Single-threaded methods
# (curvelet, channel ops) are unaffected. setdefault respects any user override.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys
import glob
import traceback

from datetime import datetime

from Dataset.update_datset import checkDataset
from Dataset.constants import kaist_path
from test_scheduler import TestQueue, stop_env_var
from test_scheduler import isTimetableActive, sleep_until

from utils import Logger, log, log_ntfy, logCoolMessage, bcolors, getGPUTestID
from Dataset import generateCFGFiles, clearCFGFIles
from utils import getGPUTestIDTag
from argument_parser import handleArguments, yolo_output_log_path, yolo_output_path_2

sys.path.append('.')
import src # Imports __init__.py defined in paralel to this script

def discover_models(group, tag):
    """Auto-discovery for `--iterations all`. Glob `group` (relative to runs/detect, or absolute) for every
    eligible trained model NOT already re-validated for `tag`, and return an ordered dict
    {abs best.pt path: path_base '<group_rel>/<rep_name>/<tag>'}. Because this runs when the queue entry
    EXECUTES (not when it was enqueued), models trained by earlier queue entries are included.
    Eligible = weights/best.pt + (results.yaml|results_reconstructed.yaml) + no EEHA_GUI_IGNORE."""
    detect = os.path.join(kaist_path, 'runs', 'detect')
    group = group.rstrip('/')
    gdir = group if os.path.isabs(group) else os.path.join(detect, group)
    group_rel = os.path.relpath(gdir, detect)
    out = {}
    for rep in sorted(glob.glob(gdir + '/*/')):
        if os.path.exists(os.path.join(rep, 'EEHA_GUI_IGNORE')):
            continue
        bp = os.path.join(rep, 'weights', 'best.pt')
        if not os.path.exists(bp):
            continue
        if not (os.path.exists(os.path.join(rep, 'results.yaml')) or
                os.path.exists(os.path.join(rep, 'results_reconstructed.yaml'))):
            continue
        if tag and glob.glob(os.path.join(rep, tag, '*', 'results.yaml')):
            continue  # already re-validated for this tag (idempotent on re-runs / resume)
        rep_name = os.path.basename(rep.rstrip('/'))
        out[bp] = f"{group_rel}/{rep_name}/{tag}" if tag else f"{group_rel}/{rep_name}"
    return out


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
    try:
        children = parent.children(recursive=True)
    except psutil.Error:
        children = []

    log(f"Total child processes: {len(children)}", bcolors.WARNING)
    # Zombie/already-dead children raise psutil.ZombieProcess / NoSuchProcess on name()/cmdline()/etc.
    # The validations already finished and wrote results before this cleanup runs, so a dead child here
    # must NEVER turn a successful run into a "FAILED TEST EXECUTION". Guard every psutil call.
    for child in children:
        try:
            log(f"PID: {child.pid}, Name: {child.name()}, Status: {child.status()}", bcolors.WARNING)
            log(f"\t· Cmdline: {child.cmdline()}", bcolors.WARNING)
            log(f"\t· CPU Time: {child.cpu_times()}", bcolors.WARNING)
            log(f"\t· Memory Info: {child.memory_info()}", bcolors.WARNING)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied) as e:
            log(f"\t· (skip child {getattr(child, 'pid', '?')}: {e})", bcolors.WARNING)

        if terminate_process:
            try:
                log(f"\t· Terminating process {child.pid}...", bcolors.WARNING)
                child.terminate()
                child.wait(timeout=10)
                log(f"\t· Process {child.pid} terminated.\n", bcolors.WARNING)
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
                log(f"\t· (could not terminate {getattr(child, 'pid', '?')}: {e})", bcolors.WARNING)

    active_threads = threading.enumerate()
    log(f"Total active threads: {len(active_threads)}", bcolors.WARNING)
    for thread in active_threads:
        log(f"\t· Thread Name: {thread.name}, Thread ID: {thread.ident}, Is daemon: {thread.daemon}", bcolors.WARNING)
    
    # log("Thread Tracebacks:", bcolors.WARNING)
    # faulthandler.dump_traceback()

if __name__ == '__main__':
    logger = Logger(yolo_output_log_path)
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
        discover_paths = None
        try:
            condition_list, option_list, model_list, opts = handleArguments(next_test)

            # --iterations all -> expand the group given in -m into the trained models present RIGHT NOW.
            if opts.iterations == 'all':
                discover_paths = discover_models(model_list[0], getattr(opts, 'result_tag', None))
                model_list = list(discover_paths.keys())
                opts.iterations = 1  # each discovered model is validated once
                log(f"[discover] {len(model_list)} eligible model(s) under '{next_test[next_test.index('-m') + 1]}' "
                    f"(tag={getattr(opts, 'result_tag', None)})")

            if discover_paths is not None and not model_list:
                # discover entry but the group is already fully re-validated -> no-op, skip dataset gen
                # (avoids regenerating test npz for nothing -- matters for slow fa/curvelet).
                log("[discover] no eligible models to validate right now; skipping dataset generation.")
                dataset_config_list = []
            elif not opts.dataset:
                checkDataset(options=option_list, dataset_format=opts.dformat,
                             rgb_eq=opts.rgb_eq, thermal_eq=opts.thermal_eq,
                             distortion_correct=opts.distortion_correct,
                             relabeling=opts.relabeling,
                             only_test=('val' in opts.run_mode and 'train' not in opts.run_mode))

                dataset_config_list = generateCFGFiles(condition_list, option_list, dataset_tag = opts.dformat)
            else:
                fallback_option = option_list[0] if option_list else 'unknown_option'
                fallback_condition = condition_list[0] if condition_list else None
                dataset_config_list = [(opts.dataset, fallback_condition, fallback_option)]

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
            for dataset, condition, option in dataset_config_list:
                for _mi, yolo_model in enumerate(model_list):
                    for index in range(opts.iterations):
                        log("--------------------------------------------------------------------------")
                        log(f"Start iteration {index+1}/{opts.iterations}")
                        iter_start = datetime.now()
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

                            pn_base = discover_paths.get(yolo_model) if discover_paths else opts.path_name
                            if pn_base is None:
                                if 'val' in opts.run_mode:
                                    path_name = "validate_" + yolo_model + "/" + test_name
                                elif 'train' in opts.run_mode:
                                    path_name = f'train_based_{yolo_model}/{test_name}'
                            else:
                                path_name = pn_base + "/" + test_name
                            path_name = path_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                        else:
                            opts.resume = False
                            path_name = resume_path 
                            resume_path = None # Reset :)
                            yolo_model = f'{yolo_output_path_2}/{path_name}/weights/last.pt'
                            log(f"Load previously executing test to continue in {path_name}")

                        test_queue.updateCurrentExecutingPath(path_name)
                        for mode in opts.run_mode:
                            # Both YoloExecution functions are added here so that variables
                            # that need to be ovewrytten take place before adding all that
                            # code
                            if mode == 'val':
                                from YoloExecution.validation_yolo import TestValidateYolo
                                TestValidateYolo(dataset, yolo_model, path_name, opts, logger.log_file_name, option)

                            elif mode == 'train':
                                from YoloExecution.train_yolo import TestTrainYolo
                                TestTrainYolo(dataset, yolo_model, path_name, opts, logger.log_file_name, option)
                        
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

                        iter_dur = str(datetime.now() - iter_start).split('.')[0]  # h:mm:ss (this single test)
                        # run type: train, or val tagged by --result-tag when present (e.g. npzfix / cross_llvip)
                        _tag = getattr(opts, 'result_tag', None)
                        run_type = 'train' if 'train' in opts.run_mode else (f'{_tag}-val' if _tag else 'val')
                        # In auto-discovery the discovered models are the units of work, so report "model i/N" and
                        # only the current model (its rep dir), not the whole list. iter_dur is this model's val time.
                        if discover_paths:
                            progress = f"model {_mi+1}/{len(model_list)}"
                            model_disp = os.path.basename(os.path.dirname(os.path.dirname(yolo_model)))  # <rep> dir
                        else:
                            progress = f"it {index+1}/{opts.iterations}"
                            model_disp = str(model_list)
                        log(f"Options executed ({progress}) in {getGPUTestIDTag()} were:\n\t· {dataset = }\n\t· model: {model_disp};\n\t· run mode: {opts.run_mode} [{run_type}]; took {iter_dur}")
                        theq_msg = f"; with thermal_eq" if opts.thermal_eq != 'none' else ""
                        rgbeq_msg = f"; with rgb_eq" if opts.rgb_eq != 'none' else ""
                        raw_msg = (f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {run_type} ({progress}) took {iter_dur} "
                                   f"— {opts.dformat}; {os.path.basename(dataset)}; {model_disp}{theq_msg}{rgbeq_msg}.")
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

            theq_msg = f"; with thermal_eq" if opts.thermal_eq != 'none' else ""
            rgbeq_msg = f"; with rgb_eq" if opts.rgb_eq != 'none' else ""
            raw_msg = f"Options failed (at index {index}/{opts.iterations}) in {getGPUTestIDTag()} were: dataset = {opts.dformat}; {os.path.basename(dataset)}; {model_list = }{theq_msg}{rgbeq_msg}."
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