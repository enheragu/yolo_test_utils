

import os

import shutil
from pathlib import Path

from utils import log, bcolors
from utils import getGPUTestID

from .constants import kaist_yolo_dataset_path, templates_cfg, condition_list_default, option_list_default, model_list_default, dataset_tags_default
from .constants import llvip_yolo_dataset_path

#################################
#       Dinamic CFG stuff       #
#################################

def generateCFGFiles(condition_list_in = None, option_list_in = None, data_path_in = None, dataset_tag = dataset_tags_default[0]):
    from jinja2 import Template
    global condition_list_default, option_list_default
    
    if 'kaist' in dataset_tag:
        datase_yolo_path = kaist_yolo_dataset_path
    elif 'llvip' in dataset_tag:
        datase_yolo_path = llvip_yolo_dataset_path
    else:
        log("[ERROR] [ConfigUtils::generateCFGFiles] Unknown dataset tag provided ({dataset_tag})", bcolors.ERROR)
    
    condition_list = condition_list_in if condition_list_in is not None else condition_list_default
    option_list = option_list_in if option_list_in is not None else option_list_default
    data_path = data_path_in if data_path_in is not None else datase_yolo_path
    
    cfg_generated_files = []
    
    id = getGPUTestID()

    tmp_cfg_path = os.getcwd() + f"/tmp_cfg{id}"
    if os.path.exists(tmp_cfg_path):
        shutil.rmtree(tmp_cfg_path)
    Path(tmp_cfg_path).mkdir(parents=True, exist_ok=True)
    
    
    with open(templates_cfg[dataset_tag]['template']) as file:
        template = Template(file.read())
        log(f"[ConfigUtils::generateCFGFiles] Generate GCF files with template from {templates_cfg[dataset_tag]['template']}")

    for condition in condition_list:
        for option in option_list:
            if 'extra' in templates_cfg[dataset_tag]:
                extra_arguments = templates_cfg[dataset_tag]['extra']
                data = template.render(condition=condition, option=option, data_path=data_path, **extra_arguments)
            else:
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
        id = getGPUTestID()
            
        tmp_cfg_path = os.getcwd() + f"/tmp_cfg{id}/"
        os.rmdir(tmp_cfg_path)
    except Exception as e:
        log(f"Catched exception removing tmf folder: {e}")