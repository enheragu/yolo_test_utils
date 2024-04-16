
#!/usr/bin/env python3
# encoding: utf-8


import os, errno
def updateSymlink(root_path, new_path):
    try:
        os.symlink(root_path, new_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(new_path)
            os.symlink(root_path, new_path)
        else:
            raise e