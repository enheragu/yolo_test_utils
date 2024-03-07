#!/usr/bin/env python3
# encoding: utf-8


import yaml
from yaml.loader import SafeLoader

################################
#      YAML parsing stuff      #
################################

def parseYaml(file_path):
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)

def dumpYaml(file_path, data):
    with open(file_path, "w+") as file:
        yaml.dump(data, file, encoding='utf-8')