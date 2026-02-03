#!/usr/bin/env python3
# encoding: utf-8

'''
    File with variables configuring path an setup info and utils
'''

import os
import sys

from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError


from utils import log, bcolors
from Dataset import dataset_tags_default, option_list_default, model_list_default, condition_list_default
from Dataset.constants import dataset_options

home = Path.home()
repo_path = f"{home}/eeha/yolo_test_utils"

yolo_output_path = f"{repo_path}/runs/detect"
yolo_output_path_2 = f"{os.getenv('HOME')}/eeha/kaist-cvpr15/runs/detect"
yolo_output_log_path = f"{repo_path}/runs/exec_log"


###################################
#     Argument input handling     #
###################################

def configArgParser():
    parser = ArgumentParser(description="Handle operations with YOLOv8, both Validation and Training. Tests will be executed iteratively from all combinations of the configurations provided (condition, option and model).")
    parser.add_argument('-c', '--condition', action='store', dest='clist', metavar='CONDITION',
                        type=str, nargs='*', default=None, choices=condition_list_default,
                        help=f"Condition from which datasets to use while training. Available options are {condition_list_default}. Usage: -c item1 item2, -c item3")
    parser.add_argument('-o', '--option', action='store', dest='olist', metavar='OPTION',
                        type=str, nargs='*', default=None, choices=option_list_default,
                        help=f"Option of the dataset to be used. Available options are {option_list_default}. Usage: -c item1 item2, -c item3")
    parser.add_argument('-m', '--model', action='store', dest='mlist', metavar='MODEL',
                        type=str, nargs='*', default=model_list_default, # choices=model_list_default,
                        help=f"Model to be used. Available options are {model_list_default}. Usage: -m item1 item2, -c item3")
    # Commented for now as it is called using env var. Is set in handleArguments to None by default
    #  parser.add_argument('-d', '--device', dest='device',
                        # default=None, choices=['cpu', '0', '1', 'None'],
                        # help="Device to run on, i.e. cuda --device '0' or --device '0,1,2,3' or --device 'cpu'.")
    parser.add_argument('-ca', '--cache', dest='cache',
                        type=str, default="ram", choices=['ram','disk'], # Disk needed to load npy images
                        help="True/ram, disk or False. Use cache for data loading. To load '.npy' or '.npz' files disk option is needed.")
    parser.add_argument('-p', '--pretrained', dest='pretrained',
                        type=bool, default=False, 
                        help="Whether to use a pretrained model.")
    parser.add_argument('-rm','--run-mode', action='store', dest='run_mode', metavar='MODE',
                        type=str, nargs='*', default=['train'], choices=['val', 'train'],
                        help="Run as validation or test mode. Available options are ['val', 'train']. Usage: -c item1 item2, -c item3")
    parser.add_argument('-path','--path-name', default=None, type=str, 
                        help="Path in which the results will be stored. If set to None a default path will be generated.")
    parser.add_argument('-test','--test-name', default=None, type=str, 
                        help="Test name. Implies changes in the path in which it will be stored. Left to 'None' for a default test name.")
    parser.add_argument('-df', '--dataset-format', dest='dformat', type=str, default=dataset_tags_default[1], choices=dataset_tags_default,
                        help=f"Format of the dataset to be generated. One of the following: {dataset_tags_default}")
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, default=None,
                        help=f"YAML file with dataset configuration")
    parser.add_argument('-it', '--iterations', dest='iterations', type=int, default=1, help='How many repetitions of this test will be performed secuencially.')
    parser.add_argument('-b', '--batch', dest='batch', type=int, default=16, help='Batch size when training.')
    parser.add_argument('-te', '--th_equalization', dest='thermal_eq',
                        type=str, default="none", choices=['none','clahe','expand'], # Equalization to thermal image: clahe, linear, no equalization..
                        help="What type of equalization is applied to the thermal channel.")
    parser.add_argument('-rgbe', '--rgb_equalization', dest='rgb_eq',
                        type=str, default="none", choices=['none','clahe','expand'], # Equalization to rgb image: clahe, linear, no equalization..
                        help="What type of equalization is applied to the rgb image.")
    
    def str2bool(v):
        # Método para interpretar cadenas como booleanas
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument('-det', '--deterministic', dest='deterministic', type=str2bool,
                    nargs='?', const=True, default=True,
                    help='Whether training process makes use of deterministic algorithms or not.')

    parser.add_argument('--distortion_correct', dest='distortion_correct', type=str2bool,
                    nargs='?', const=True, default=True,
                    help='Whether the dataset generation will update images with distortion correction or not.')
    parser.add_argument('--relabeling', dest='relabeling', type=str2bool,
                    nargs='?', const=True, default=True,
                    help='Whether the training will take into account the relabeling performed to the original dataset, or just the original labels.')
    return parser

def handleArguments(argument_list = sys.argv[1:]):
    arg_dict = {}
    
    parser = configArgParser()
    opts = parser.parse_args(argument_list)

    condition_list = None if opts.clist is None else list(opts.clist)
    option_list = None if opts.olist is None else list(opts.olist)
    model_list = None if opts.mlist is None else list(opts.mlist)
    run_modes = None if opts.run_mode is None else list(opts.run_mode)

    # Auto-configure cache to 'disk' if any selected option uses .npz or .npy extension
    if option_list and opts.cache == 'ram':
        for opt in option_list:
            if opt in dataset_options:
                ext = dataset_options[opt].get('extension', '')
                if ext in ['.npz', '.npy']:
                    log(f"Option '{opt}' uses extension '{ext}', switching cache from 'ram' to 'disk'.")
                    opts.cache = 'disk'
                    break

    if not opts.dataset:
        if opts.dformat not in dataset_tags_default:
            raise KeyError(f"Dataset format provided ({opts.dformat}) is not part of the ones avalable: {dataset_tags_default}.")

    opts.device = None
    if "EEHA_TRAIN_DEVICE" in os.environ:
        device_value = os.getenv("EEHA_TRAIN_DEVICE")
        if device_value.isdigit():
            device_number = int(device_value)
            log(f"Device has been configured through ENV(EEHA_TRAIN_DEVICE) to {device_number}. Argparse had previously set it to {opts.device}.")
            opts.device = device_number
        else:
            raise TypeError(f"Invalid device set thorugh ENV(EEHA_TRAIN_DEVICE). Is not a number: {device_number}")

    if "EEHA_ACTIVE_TEST_TIMETABLE" in os.environ:
        timetable = str(os.getenv("EEHA_TRAIN_DEVICE"))
        log(f"Timetable configured, tests will only be executed between {timetable} hours")

    log(f"Options parsed:\n\t· condition_list: {condition_list}\n\t· option_list: {option_list}\n\t· model_list: {model_list};\n\t· run mode: {run_modes}")
    log(f"Extra options are: {opts}")
    return condition_list, option_list, model_list, opts
