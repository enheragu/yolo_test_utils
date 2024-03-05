#!/usr/bin/env python3
# encoding: utf-8

"""
    This file defines a class to handle scheduling different test options to be later executed.
    Makes use of different set of files to store these tests and of a FileLock class to handle 
    safe access to them.

    Each test stored is composed by a list of options that will be later retrieved and executed.

    When executed as a main file (./test_scheduler.py) it tries to add given options (if provided)
    to the pending list.
"""

import os
import sys
import fcntl
import yaml
import shutil

from log_utils import log, bcolors

sys.path.append('.')
import src # Imports __init__.py defined in paralel to this script

# Important path and file names used by this module
cache_path = f"{os.getenv('HOME')}/.cache/eeha_yolo_test"
pending_file_default = f'{cache_path}/pending.yaml'
pending_stopped_default = f'{cache_path}/pending_stopped.yaml'

id = ""
if "EEHA_TRAIN_DEVICE" in os.environ:
    id = f'_GPU{os.getenv("EEHA_TRAIN_DEVICE")}'
executing_file_default = f'{cache_path}/executing{id}.yaml'
finished_file_ok_default = f'{cache_path}/finished_ok.yaml'
finished_file_failed_default = f'{cache_path}/finished_failed.yaml'
stop_env_var = "EEHA_TEST_STOP_REQUESTED"

"""
    Class that handles safe lock/unlock mechanism for files. It is set
    so that it locks when created and unlocks when the environment is
    left and the variable is destroyed
"""

class FileLock:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)


class TestQueue:
    def __init__(self, pending_file = pending_file_default,
                       pending_stopped_file = pending_stopped_default, 
                       executing_file = executing_file_default, 
                       finished_file_ok = finished_file_ok_default,
                       finished_file_failed = finished_file_failed_default,
                       reset_files = False):
        
        self.pending_file = pending_file
        self.pending_stopped_file = pending_stopped_file
        self.executing_file = executing_file
        self.finished_file_ok = finished_file_ok
        self.finished_file_failed = finished_file_failed

        ## Reset previous stop request if any
        if os.getenv("EEHA_TEST_STOP_REQUESTED"):
            os.environ.pop("EEHA_TEST_STOP_REQUESTED")
        
        # Create cache path if it does not exist
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        
        if reset_files and os.path.exists(cache_path):
            shutil.rmtree(cache_path)

        self.executing_test = None

    def _read_file(self, file_name):
        try:
            with open(file_name, 'r') as file:
                return yaml.safe_load(file) or []
        except FileNotFoundError:
            return []
    
    def _save_file(self, file_name, data):
        with open(file_name, 'w+') as file:
            yaml.safe_dump(data, file)

    def _updateFile(self, file_name, new_data):
        FileLock(file_name)
        data = self._read_file(file_name)
        data.append(new_data)
        self._save_file(file_name, data)

    def _handleStoppedTests(self):
        FileLock(self.pending_file)
        FileLock(self.pending_stopped_file)
        pending = self._read_file(self.pending_file)
        self._save_file(self.pending_file, [])
        self._save_file(self.pending_stopped_file, pending)

    def _popFirst(self, file_name):
        FileLock(file_name)
        items = self._read_file(file_name)
        next_test = None
        if items:
            next_test = items.pop(0)
            self._save_file(file_name, items)
        return next_test

    """
        Interface method to get next test to execute
    """
    def get_next_test(self):

        if os.getenv("EEHA_TEST_STOP_REQUESTED"):
            log("Env EEHA_TEST_STOP_REQUESTED detected. Stopping execution.")
            self._handleStoppedTests()
            return None
        
        next_test = self._popFirst(self.pending_file)

        FileLock(self.executing_file)
        self._save_file(self.executing_file, [next_test])
        self.executing_test = next_test

        return next_test

    """
        Interface method to notify and log the test that was already executed,
        with success status (true or false)
    """
    def finished_test(self, success = True):
        if self.executing_test:
            if success:
                self._updateFile(self.finished_file_ok, self.executing_test)
            else:
                self._updateFile(self.finished_file_failed, self.executing_test)
        
        FileLock(self.executing_file)
        self._save_file(self.executing_file, [])

    """
        Interface method to add new tests to queue
    """
    def add_new_test(self, new_test):
        self._updateFile(self.pending_file, new_test)

## TEST MODULE
def test():
    test_queue = TestQueue()
    print(f"[TEST] Add to test queue: {sys.argv[1:]}")
    test_queue.add_new_test(sys.argv[1:])
    print(f"[TEST] Add again to test queue: {sys.argv[1:]}")
    test_queue.add_new_test(sys.argv[1:])
    test_queue.add_new_test(sys.argv[1:])
    test_queue.add_new_test(sys.argv[1:])
    test_queue.add_new_test(sys.argv[1:])
    test_queue.add_new_test(sys.argv[1:])

    next_test = test_queue.get_next_test()
    print(f"[TEST] Got test queue: {next_test}")
    print(f"[TEST] Finish ok: {next_test}")
    test_queue.finished_test(True)

    next_test = test_queue.get_next_test()
    print(f"[TEST] Finish no ok: {next_test}")
    test_queue.finished_test(False)

    next_test = test_queue.get_next_test()
    
    os.environ[stop_env_var] = "True"
    next_test = test_queue.get_next_test()

    test_queue = TestQueue(reset_files=True)
    
    
if __name__ == '__main__':
    test_queue = TestQueue()

    if len(sys.argv) > 1:
        log(f"Add new test to queue: {sys.argv[1:]}")
        test_queue.add_new_test(sys.argv[1:]) 
    else:
        log(f"Not enough arguments provided to add new test", bcolors.ERROR)