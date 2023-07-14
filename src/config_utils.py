'''
    File with variables configuring path an setup info and utils
'''

import os
from pathlib import Path


home = Path.home()
repo_path = f"{home}/eeha/yolo_test_utils/"

kaist_path = f"{home}/eeha/kaist-cvpr15"
sets_path = f"{kaist_path}/imageSets/"
annotation_path = f"{kaist_path}/annotations-xml-new/"
images_path = f"{kaist_path}/images/"
yolo_dataset_path = f"{home}//eeha/kaist-yolo-annotated/"

dataset_config_path = f"{repo_path}/yolo_config/"
cfg_template = f"{dataset_config_path}/dataset_condition_option.j2"
yolo_architecture_path = f"{dataset_config_path}/yolo_eeha_n.yaml"
yolo_output_path = f"{repo_path}/runs/detect"


#################################
#       Dinamic CFG stuff       #
#################################

condition_list = ['all'] #('day', 'night', 'all')
option_list = ['visible', 'lwir', 'hsvt', 'rgbt']


def generateCFGFiles(condition_list = condition_list, option_list = option_list, data_path = yolo_dataset_path):
    from jinja2 import Template
    
    cfg_generated_files = []
    
    tmp_cfg_path = os.getcwd() + "/tmp_cfg/"
    Path(tmp_cfg_path).mkdir(parents=True, exist_ok=True)

    with open(cfg_template) as file:
        template = Template(file.read())

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
        print(f"Catched exception removing tmf folder: {e}")



################################
#     Format Logging stuff     #
################################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log(msg = "", color = bcolors.OKCYAN):
    print(f"{color}{msg}{bcolors.ENDC}")

generateCFGFiles()