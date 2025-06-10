
#!/usr/bin/env python3
# encoding: utf-8


from functools import wraps
import time

import numpy as np
import cv2 as cv

def save_image_if_path(func):
    @wraps(func)
    def wrapper(*args, path=None, **kwargs):
        result = func(*args, **kwargs)
        if path is not None and isinstance(result, np.ndarray):
            cv.imwrite(path, result)
            return None
        return result
    return wrapper

def save_npmat_if_path(func):
    @wraps(func)
    def wrapper(*args, path=None, **kwargs):
        result = func(*args, **kwargs)
        if path is not None:
            if isinstance(result, np.ndarray):
                # np.save(path.replace('.png',''), result)
                np.savez_compressed(path.replace('.png','').replace('.jpg',''), image = result)    
                return None
            else:
                raise ValueError("[Decorators] Result is not a numpy array, cannot save as npz.")
        return result
    return wrapper

# Decorator that measures time execution
def time_execution_measure(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return result, total_time
    return timeit_wrapper