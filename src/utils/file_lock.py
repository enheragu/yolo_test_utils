import os
import fcntl
import time

class FileLock:
    def __init__(self, filename):
        self.filename = filename
        
        self.acquire()
        
    def acquire(self):
        
        while os.path.exists(self.filename):
            time.sleep(0.5)
            
        # print('[FileLock::acquire]')
        file = open(self.filename, "w")
        file.write(":)")
        file.close()
        return file
    
    def __enter__(self):
        # Needed to work with 'with' context
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Needed to work with 'with' context
        self.release()
        
    def release(self):
        # print('[FileLock::release] - Released lock and closed file')
        if os.path.exists(self.filename):
            os.remove(self.filename)
        
    def __del__(self):
        self.release()


