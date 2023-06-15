'''
    File with variables configuring path an setup info
'''


kaist_path = "/home/arvc/eeha/kaist-cvpr15"
sets_path = f"{kaist_path}/imageSets/"
annotation_path = f"{kaist_path}/annotations-xml-new/"
images_path = f"{kaist_path}/images/"
yolo_dataset_path = "/home/arvc/eeha/kaist-yolo-annotated/"

dataset_config_yaml = 'yolo_config/yolo_dataset.yaml'
yolo_architecture_path = 'yolo_config/yolo_eeha_n.yaml'
yolo_val_output = "/home/quique/umh/yolo/ultralitics_yolov8/runs/detect/"