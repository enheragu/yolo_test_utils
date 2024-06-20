
import os
import fcntl

"""
    Class that handles safe lock/unlock mechanism for files. It is set
    so that it locks when created and unlocks when the environment is
    left and the variable is destroyed
"""

class FileLock:
    def __init__(self, file):
        self.file = file
        if not os.path.exists(self.file): # Create file and lock
            with open(self.file, 'w+'):
                pass

    def __enter__(self):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)