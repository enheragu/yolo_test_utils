'''
    File with variables configuring path an setup info and utils
'''

import os
from pathlib import Path

from datetime import datetime

home = Path.home()
repo_path = f"{home}/eeha/yolo_test_utils"

dataset_config_path = f"{repo_path}/yolo_config"
yolo_output_path = f"{repo_path}/runs/detect"

####################################
#     Datasets path and output     #
####################################

kaist_path = f"{home}/eeha/kaist-cvpr15"
kaist_sets_path = f"{kaist_path}/imageSets"
kaist_annotation_path = f"{kaist_path}/annotations-xml-new"
kaist_images_path = f"{kaist_path}/images"
kaist_yolo_dataset_path = f"{home}/eeha/kaist-yolo-annotated" # Output dataset in YOLO format

templates_cfg = {'kaist_coco': f"{dataset_config_path}/dataset_kaist_coco_option.j2",
                 'kaist': f"{dataset_config_path}/dataset_kaist_option.j2"
                 }


################################
#     Format Logging stuff     #
################################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log(msg = "", color = bcolors.OKCYAN):
    timetag = datetime.utcnow().strftime('%F %T.%f')[:-3]
    print(f"{color}[{timetag}] {msg}{bcolors.ENDC}")


################################
#      YAML parsing stuff      #
################################

import yaml
from yaml.loader import SafeLoader
def parseYaml(file_path):
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)

def dumpYaml(file_path, data):
    with open(file_path, "w+") as file:
        yaml.dump(data, file)
        
#################################
#       Dinamic CFG stuff       #
#################################
import shutil

condition_list_default = ['all','day', 'night']
option_list_default = ['visible', 'lwir', 'hsvt', 'rgbt', 'vths', 'vt', '4ch'] # 4ch
model_list_default = ['yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
dataset_tags_default = ['kaist_coco', 'kaist'] # Just list of availables :)

def generateCFGFiles(condition_list_in = None, option_list_in = None, data_path_in = None, dataset_tag = dataset_tags_default[0]):
    from jinja2 import Template

    global condition_list_default, option_list_default, kaist_yolo_dataset_path
    
    condition_list = condition_list_in if condition_list_in is not None else condition_list_default
    option_list = option_list_in if option_list_in is not None else option_list_default
    data_path = data_path_in if data_path_in is not None else kaist_yolo_dataset_path
    
    cfg_generated_files = []
    
    tmp_cfg_path = os.getcwd() + "/tmp_cfg"
    if os.path.exists(tmp_cfg_path):
        shutil.rmtree(tmp_cfg_path)
    Path(tmp_cfg_path).mkdir(parents=True, exist_ok=True)
    
    
    with open(templates_cfg[dataset_tag]) as file:
        template = Template(file.read())
        log(f"[ConfigUtils::generateCFGFiles] Generate GCF files with template from {templates_cfg[dataset_tag]}")

    for condition in condition_list:
        for option in option_list:
            data = template.render(condition=condition, option=option, data_path=data_path)

            file_path = f"{tmp_cfg_path}/dataset_{condition}_{option}.yaml"
            with open(file_path, mode='w') as f:
                f.write(data)
            cfg_generated_files.append(file_path)

    return cfg_generated_files
    
def clearCFGFIles(cfg_generated_files):
    # Note that if files are not cleared, rmdir will not work as folder would not be empty
    
    for file in cfg_generated_files:
        if os.path.isfile(file):
            os.remove(file)
    
    try:
        tmp_cfg_path = os.getcwd() + "/tmp_cfg/"
        os.rmdir(tmp_cfg_path)
    except Exception as e:
        log(f"Catched exception removing tmf folder: {e}")


###################################
#     Argument input handling     #
###################################
from argparse import ArgumentParser

def handleArguments():
    global condition_list_default, option_list_default, model_list_default
    arg_dict = {}
    parser = ArgumentParser(description="Handle operations with YOLOv8, both Validation and Training. Tests will be executed iteratively from all combinations of the configurations provided (condition, option and model).")
    parser.add_argument('-c', '--condition', action='store', dest='clist', metavar='CONDITION',
                        type=str, nargs='*', default=condition_list_default,
                        help=f"Condition from which datasets to use while training. Available options are {condition_list_default}. Usage: -c item1 item2, -c item3")
    parser.add_argument('-o', '--option', action='store', dest='olist', metavar='OPTION',
                        type=str, nargs='*', default=option_list_default,
                        help=f"Option of the dataset to be used. Available options are {option_list_default}. Usage: -c item1 item2, -c item3")
    parser.add_argument('-m', '--model', action='store', dest='mlist', metavar='MODEL',
                        type=str, nargs='*', default=model_list_default,
                        help=f"Model to be used. Available options are {model_list_default}. Usage: -c item1 item2, -c item3")
    parser.add_argument('-d', '--device', dest='device',
                        default=None, 
                        help="Device to run on, i.e. cuda --device '0' or --device '0,1,2,3' or --device 'cpu'.")
    parser.add_argument('-ca', '--cache', dest='cache',
                        type=str, default="ram",
                        help="True/ram, disk or False. Use cache for data loading. To load '.npy' or '.npz' files disk option is needed.")
    parser.add_argument('-p', '--pretrained', dest='pretrained',
                        type=bool, default=False, 
                        help="Whether to use a pretrained model.")
    parser.add_argument('-rm','--run-mode', action='store', dest='run_mode', metavar='MODE',
                        type=str, nargs='*', default=['val', 'train'],
                        help="Run as validation or test mode. Available options are ['val', 'train']. Usage: -c item1 item2, -c item3")
    parser.add_argument('-path','--path-name', default=None, type=str, 
                        help="Path in which the results will be stored. If set to None a default path will be generated.")
    parser.add_argument('-df', '--dataset-format', dest='dformat', type=str, default=dataset_tags_default[0],
                        help=f"Format of the dataset to be generated. One of the following: {dataset_tags_default}")

    
    opts = parser.parse_args()

    condition_list_default = list(opts.clist)
    option_list_default = list(opts.olist)
    model_list_default = list(opts.mlist)
    run_modes = list(opts.run_mode)

    if opts.dformat not in dataset_tags_default:
        raise KeyError(f"Dataset format provided ({opts.dformat}) is not part of the ones avalable: {dataset_tags_default}.")

    log(f"Options parsed:\n\t· condition_list: {condition_list_default}\n\t· option_list: {option_list_default}\n\t· model_list: {model_list_default};\n\t· run mode: {run_modes}")
    return condition_list_default, option_list_default, model_list_default, opts
