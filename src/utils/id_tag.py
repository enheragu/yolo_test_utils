#!/usr/bin/env python3
# encoding: utf-8

import os

def getGPUTestIDTag():
    id = []

    if "EEHA_TRAIN_DEVICE_TAG" in os.environ:
        id.append(os.getenv("EEHA_TRAIN_DEVICE_TAG"))

    if "EEHA_TRAIN_DEVICE" in os.environ:
        id.append(f'GPU{os.getenv("EEHA_TRAIN_DEVICE")}')

    return '_'.join(id)

# Returns an STR tag to ID current test execution with _
def getGPUTestID():
    id_str = getGPUTestIDTag()
    if id_str:
        id_str = '_' + id_str 
    return id_str
