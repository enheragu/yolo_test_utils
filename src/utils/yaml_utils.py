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

# Check if all values in the dictionary/list are of basic types
def is_basic_types(values):
    basic_types = (int, str, bool, float)
    return all(isinstance(value, basic_types) for value in values)

# Custom representation function for dictionaries
def represent_dict(dumper, data):
    if is_basic_types(data.values()):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data)
    else:
        return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=False)

# Custom representation function for lists
def represent_list(dumper, data):
    if is_basic_types(data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    else:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

def dumpYaml(file_path, data, mode = "w+"):
    with open(file_path, mode) as file:
        # Add custom representation functions to the YAML dumper
        yaml.add_representer(list, represent_list)
        yaml.add_representer(dict, represent_dict)
        yaml.dump(data, file, encoding='utf-8', width=float(3000))

