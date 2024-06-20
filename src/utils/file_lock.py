
import os
import fcntl
import time

"""
    Class that handles safe lock/unlock mechanism for files. It is set
    so that it locks when created and unlocks when the environment is
    left and the variable is destroyed
"""

class FileLock:
    def __init__(self, file, blocking = False):
        self.file = file
        self.blocking = blocking

    def __enter__(self):

        if not os.path.exists(self.file): # Create file and lock
            with open(self.file, 'w+') as file:
                file.write(':)')

        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)

        while True:
            try:
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB))
                break  # Si se obtiene el bloqueo, salir del bucle
            except BlockingIOError:
                if not self.blocking:
                    return
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)