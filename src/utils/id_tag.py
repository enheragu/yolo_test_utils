#!/usr/bin/env python3
# encoding: utf-8

import os

# Returns an STR tag to ID current test execution
def getGPUTestID():
    id = ""
    if "EEHA_TRAIN_DEVICE_TAG" in os.environ:
        id += f'_{os.getenv("EEHA_TRAIN_DEVICE_TAG")}'

    if "EEHA_TRAIN_DEVICE" in os.environ:
        id += f'_GPU{os.getenv("EEHA_TRAIN_DEVICE")}'
    return id