import numpy as np
import os
import shutil
import mujoco
import time

def create_or_empty_dir(directory):
        if os.path.exists(directory):
            # Remove all files in the directory
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            os.makedirs(directory)

def print_debug(x,debug_mode,module_id=None):
    if debug_mode:
        if module_id is not None:
            print(f"[{module_id} MODULE]: {x}")
        else:
            print(x)

def sleep(seconds,sleep=True):
    if sleep:
        time.sleep(seconds)