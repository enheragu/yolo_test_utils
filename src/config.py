'''
    File with variables configuring path an setup info
'''


kaist_path = "/home/arvc/eeha/kaist-cvpr15"
sets_path = f"{kaist_path}/imageSets/"
annotation_path = f"{kaist_path}/annotations-xml-new/"
images_path = f"{kaist_path}/images/"
yolo_dataset_path = "/home/arvc/eeha/kaist-yolo-annotated/"

dataset_config_yaml = 'yolo_config/yolo_dataset.yaml'
dataset_config_path = 'yolo_config/'
dataset_config_list = ('dataset_night_visible.yaml',
                       'dataset_night_lwir.yaml',
                       'dataset_day_visible.yaml',
                       'dataset_day_lwir.yaml')

yolo_architecture_path = 'yolo_config/yolo_eeha_n.yaml'
yolo_val_output = "/home/arvc/eeha/yolo_test_utils/runs/detect"



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


def log(msg = ""):
    print(f"{bcolors.OKCYAN}{msg}{bcolors.ENDC}")
