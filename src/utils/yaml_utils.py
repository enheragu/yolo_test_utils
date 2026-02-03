#!/usr/bin/env python3
# encoding: utf-8


import yaml
from yaml.loader import SafeLoader
from argparse import Namespace

################################
#      YAML parsing stuff      #
################################

def ns_constructor(loader, node):
    data = loader.construct_mapping(node)
    return Namespace(**data)

yaml.add_constructor(
    u'tag:yaml.org,2002:python/object:argparse.Namespace',
    ns_constructor,
    Loader=yaml.SafeLoader
)

def parseYaml(file_path):

    try:
        with open(file_path) as file:
            data = yaml.load(file, Loader=SafeLoader)
            if isinstance(data, Namespace):
                data = vars(data)
            return data
    except yaml.YAMLError as exc:
        print(f"Error in YAML file: {exc}")
        
    return {}

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
        yaml.dump(data, file, encoding='utf-8', width=float(5000))

