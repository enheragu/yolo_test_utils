#!/usr/bin/env python3
# encoding: utf-8

"""
    Kaist to Yolo annotation formatting: one *.txt file per image (if no objects in image, no *.txt file is required). 
    The *.txt file specifications are:
    · One row per object
    · Each row is class x_center y_center width height format.
    · Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and 
      width by image width, and y_center and height by image height.
    · Class numbers are zero-indexed (start from 0).
"""
import untangle
import yaml
import os, errno
from pathlib import Path
import shutil

from multiprocessing.pool import Pool
from functools import partial

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')

from config_utils import kaist_sets_path, kaist_annotation_path, kaist_images_path, kaist_yolo_dataset_path, log

dataset_tag = 'kaist_coco' # 'kaist'
# 'kaist_coco' is a version of kais that makes use of compatible coco tags
# 'kaist' is regular kaist with its tagged classes

lwir = "/lwir/"
visible = "/visible/"

label_folder = "/labels/"
images_folder = "/images/"


# TO check against default yolo, classes have to match the coco128.yaml
class_data_coco = {  'person': 0,  'cyclist': 80, 'people': 81 } # people does not exist in coco dataset, use 80 as tag

def processXML(xml_path, output_paths):
    global dataset_tag
    global class_data_coco

    index = 0
    if dataset_tag == 'kaist_coco':
        obj_class_dict = class_data_coco
    else:
        obj_class_dict = {}

    with open(xml_path) as xml:
        txt_data = ""
        doc = untangle.parse(xml)
        if hasattr(doc.annotation, "object"):
            for object in doc.annotation.object:
                obj_name = object.name.cdata.replace("?","")
                                
                img_width = float(doc.annotation.size.width.cdata)
                img_height = float(doc.annotation.size.height.cdata)

                x_centered = float(object.bndbox.x.cdata) + float(object.bndbox.w.cdata) / 2.0
                y_centered = float(object.bndbox.y.cdata) + float(object.bndbox.h.cdata) / 2.0

                x_normalized = x_centered / img_width
                y_normalized = y_centered / img_height
                w_normalized = float(object.bndbox.w.cdata) / img_width
                h_normalized = float(object.bndbox.h.cdata) / img_height
                
                if dataset_tag == 'kaist_coco':
                    if obj_name == "people" or obj_name == "cyclist":
                        obj_name = "person"
                        
                    # Only processes person for now 
                    if obj_name == "person":
                        txt_data += f"{obj_class_dict[obj_name]} {x_normalized} {y_normalized} {w_normalized} {h_normalized}\n"

                elif dataset_tag == 'kaist':
                        if obj_name not in obj_class_dict:
                            obj_class_dict[obj_name] = index
                        txt_data += f"{obj_class_dict[obj_name]} {x_normalized} {y_normalized} {w_normalized} {h_normalized}\n"

                index += 1
            for file in output_paths:
                with open(file, 'w+') as output:
                    output.write(txt_data)
        log("Class dict is now: {class_dict}")
# Process line from dataset file so to paralelice process
def processLine(new_dataset_label_paths, data_set_name, line):
    for data_type in (lwir, visible):
        line = line.replace("\n", "")
        path = line.split("/")
        path = (path[0], path[1], data_type, path[2])
        root_label_path = kaist_annotation_path + line + ".xml"
        # log(root_label_path)

        # labelling

        root_label_path = f"{kaist_annotation_path}/{line}.xml"
        output_paths = [f"{folder}/{path[0]}_{path[1]}_{path[3]}.txt" for folder in  new_dataset_label_paths]
        # log(output_paths)
        processXML(root_label_path, output_paths)

        # Create images
        root_image_path = kaist_images_path + "/".join(path) + ".jpg"
        new_image_path = f"{kaist_yolo_dataset_path}{data_set_name}{data_type}{images_folder}{path[0]}_{path[1]}_{path[3]}.png"
        # log(new_image_path)
        # Create or update symlink if already exists
        try:
            os.symlink(root_image_path, new_image_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(new_image_path)
                os.symlink(root_image_path, new_image_path)
            else:
                raise e
            
dataset_blacklist = []
dataset_whitelist = ['train-all-02', 'train-all-20', 'test-all-01', 'train-day-04', 'train-day-20', 'test-day-01', 'test-day-20', 'train-night-02', 'train-night-04', 'test-night-01', 'test-night-20']
def kaistToYolo(dataset_format = 'kaist_coco'):
    global dataset_tag
    dataset_tag = dataset_format

    dataset_processed = 0
    # Goes to imageSets folder an iterate through the images an processes all image sets
    log("[KaistToYolo::KaistToYolo] Kaist To Yolo formatting in {dataset_format} format:")
    for file in os.listdir(kaist_sets_path):
        file_path = os.path.join(kaist_sets_path, file)
        if os.path.isfile(file_path):
            data_set_name = file.replace(".txt", "")

            # Check that is not empty
            if data_set_name not in dataset_whitelist and dataset_whitelist:
                log(f"\t· Dataset {data_set_name} is not in whitelist. Not processed")
                continue
            elif data_set_name in dataset_blacklist:
                log(f"\t· Dataset {data_set_name} is in blacklist. Not processed")
                continue

            # log(f"\t[{dataset_processed}] Processing dataset {data_set_name}")

            # Create new folder structure
            new_dataset_label_paths = (kaist_yolo_dataset_path + data_set_name + lwir + label_folder,
                                kaist_yolo_dataset_path + data_set_name + visible + label_folder)
            new_dataset_kaist_images_paths = (kaist_yolo_dataset_path + data_set_name + lwir + images_folder,
                                kaist_yolo_dataset_path + data_set_name + visible + images_folder)
            
            for folder in (new_dataset_label_paths + new_dataset_kaist_images_paths):
                # log(folder)
                Path(folder).mkdir(parents=True, exist_ok=True)

            # Process all lines in imageSet file to create labelling in the new folder structure
            with open(file_path, 'r') as file:
                lines_list = [line.rstrip() for line in file]
                with Pool() as pool:
                    func = partial(processLine, new_dataset_label_paths, data_set_name)
                    pool.map(func, lines_list)
                    
                log(f"\t· [{dataset_processed}] Processed {data_set_name} dataset: {len(lines_list)} XML files (and x2 images: visible and lwir) in {data_set_name} dataset")
            dataset_processed += 1
                        
    log(f"[KaistToYolo::KaistToYolo] Finished procesing {dataset_processed} datasets. Output datasests are located in {kaist_yolo_dataset_path}")
    
    # yaml_data_path = "./dataset_config/yolo_obj_classes.yaml"
    # with open(yaml_data_path, "w+") as file:
    #     # Swap key and value to access by number later
    #     yaml_data = {"path": kaist_images_path, "train": "#TBD", "val": "#TBD", "test": "#TBD",
    #                 "names": {v: k for k, v in class_data_coco.items()}}
    #     yaml.dump(yaml_data, file)

    # log(f"Dumped data about classes in: {yaml_data_path}. \nData is: \t\n{yaml_data}")
    # log(f"Processed files: {processed_files}")


if __name__ == '__main__':
    kaistToYolo()