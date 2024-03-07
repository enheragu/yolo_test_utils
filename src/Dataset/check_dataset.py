
#!/usr/bin/env python3
# encoding: utf-8

import os
from utils import log, bcolors

from .constants import images_folder_name, labels_folder_name

"""
    Check that for each image theres a labels file in the corresponding folder to avoid
    later execution errors
"""
def checkImageLabelPairs(path):
    image_directories = []
    for root, dirs, files in os.walk(path):
        if images_folder_name in dirs:
            images_folder = os.path.join(root, images_folder_name)
            image_directories.append(images_folder)
    # log(f"\t · Check image/label correspondance in {path}.")
    for images_dir in image_directories:
        labels_dir = images_dir.replace(images_folder_name, labels_folder_name)
        image_files = set([os.path.splitext(file)[0] for file in os.listdir(images_dir)])
        label_files = set([os.path.splitext(file)[0] for file in os.listdir(labels_dir)])
        
        # Verificar si los archivos en 'images' tienen sus equivalentes en 'labels'
        missing_labels = image_files - label_files
        extra_labels = label_files - image_files

        if not missing_labels and not extra_labels:
            # log(f"Okey in {images_dir}", bcolors.OKGREEN)
            pass
        else:
            # Missing labels is not a problem as YOLO can interpret that no label is needed for that image
            # if missing_labels:
            #     log(f"Missing labels in {labels_dir}: {len(missing_labels)} missing labels. Num images: {len(image_files)}; Num labels: {len(label_files)}", bcolors.ERROR)
            if extra_labels:
                log(f"\t · Labels without corresponding image in {images_dir}: {len(extra_labels)} extra labels  |  Num images: {len(image_files)}; Num labels: {len(label_files)}", bcolors.ERROR)
                