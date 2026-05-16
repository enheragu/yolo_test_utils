#!/usr/bin/env python3
# encoding: utf-8
"""
    Single source of truth for path configurations.
    This module has NO heavy dependencies to allow lightweight imports.
"""

import os
from pathlib import Path

home = Path.home()
repo_path = f"{home}/eeha/yolo_test_utils"

# YOLO output paths
yolo_output_path = f"{repo_path}/runs/detect"
yolo_output_path_2 = f"{os.getenv('HOME')}/eeha/kaist-cvpr15/runs/detect"
yolo_output_log_path = f"{repo_path}/runs/exec_log"
