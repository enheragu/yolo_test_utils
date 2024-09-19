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
import os
from pathlib import Path

from multiprocessing.pool import Pool
from functools import partial

import cv2 as cv

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    sys.path.append('./src/Dataset')

from utils import updateSymlink
from Dataset.constants import class_data, dataset_whitelist, dataset_blacklist, kaist_sets_paths, kaist_annotation_path, kaist_images_path, kaist_yolo_dataset_path
from Dataset.constants import images_folder_name, labels_folder_name, lwir_folder_name, visible_folder_name
from Dataset.th_equalization import th_equalization, rgb_equalization
# from .check_dataset import checkImageLabelPairs

from utils import log, bcolors


def processXML(xml_path, output_paths, dataset_format):
    global class_data
    global obj_class_dict

    obj_class_dict = class_data[dataset_format]

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
                
                if dataset_format == 'kaist_coco':
                    if obj_name == "people" or obj_name == "cyclist":
                        obj_name = "person"
                        
                    # Only processes person for now 
                    if obj_name == "person":
                        txt_data += f"{obj_class_dict[obj_name]} {x_normalized} {y_normalized} {w_normalized} {h_normalized}\n"

                # For now Kaist format takes only into account persons
                # Assumes kaist regular format
                # (dataset_format == 'kaist_small' or dataset_format == 'kaist_full')
                elif obj_name == "person":
                        txt_data += f"{obj_class_dict[obj_name]} {x_normalized} {y_normalized} {w_normalized} {h_normalized}\n"

            # for file in output_paths:
            if len(output_paths) >=3:
                log(f"len(output_paths) >=3 - {len(output_paths) = }: {output_paths}", bcolors.ERROR)
            
            # First file made, second is symlink to first one
            with open(output_paths[0], 'w+') as output:
                output.write(txt_data)
            
            updateSymlink(output_paths[0], output_paths[1])

# Process line from dataset file so to paralelice process
## IMPORTANT -> line has to be the last argument
def processLineLabels(new_dataset_label_paths, dataset_format, line):
    path = '_'.join(line.split("/"))
    
    root_label_path = os.path.join(kaist_annotation_path,f"{line}.xml")
    output_paths = [os.path.join(folder,f"{path}.txt") for folder in  new_dataset_label_paths]
    # log(output_paths)
    processXML(root_label_path, output_paths, dataset_format)


def processLineImages(data_set_name, rgb_eq, thermal_eq, line):
    processed = {lwir_folder_name: {}, visible_folder_name: {}}

    for data_type in (lwir_folder_name, visible_folder_name):
        path = line.split("/")
        path = (path[0], path[1], data_type, path[2])

        # Create images
        root_image_path = os.path.join(kaist_images_path,"/".join(path) + ".jpg")
        new_image_path = os.path.join(kaist_yolo_dataset_path,data_set_name,data_type,images_folder_name,f"{path[0]}_{path[1]}_{path[3]}.png")

        # Apply clahe equalization to LWIR images if needed, or add symlink instead
        if 'lwir' in data_type and str(thermal_eq).lower() != 'none':
            th_img = cv.imread(root_image_path, cv.IMREAD_GRAYSCALE) # It is enconded as BGR so still needs merging to Gray
            th_img = th_equalization(th_img, thermal_eq)
            cv.imwrite(new_image_path, th_img)
        elif str(rgb_eq).lower() != 'none':
            # log(new_image_path)
            # Create or update symlink if already exists
            # updateSymlink(root_image_path, new_image_path)
            rgb_img = cv.imread(root_image_path) # It is enconded as BGR so still needs merging to Gray
            rgb_img = rgb_equalization(rgb_img, rgb_eq)
            cv.imwrite(new_image_path, rgb_img)
        else:
            # Create or update symlink if already exists
            updateSymlink(root_image_path, new_image_path)
        
        processed[data_type][line] = new_image_path
    
    return processed
    # log(f"[KaistToYolo::processLine] Process {root_label_path}")


def upateProcessedSymlinks(pre_processed, data_set_name, line):
    for data_type in (lwir_folder_name, visible_folder_name):
        img_root_path = pre_processed[data_type][line]
        img_new_path = img_root_path.replace(img_root_path.split("/")[-4], data_set_name)
        updateSymlink(img_root_path, img_new_path)

        label_root_path = pre_processed[data_type][line].replace(images_folder_name, labels_folder_name).replace('.png', '.txt')
        label_new_path = label_root_path.replace(label_root_path.split("/")[-4], data_set_name)
        # Check if there is labels root path or is just a background image
        if os.path.exists(label_root_path):
            updateSymlink(label_root_path, label_new_path)
    

def kaistToYolo(dataset_format = 'kaist_coco', rgb_eq = 'none', thermal_eq = 'none'):
    global class_data

    dataset_processed = 0
    pre_processed = {lwir_folder_name: {}, visible_folder_name: {}}
    # Goes to imageSets folder an iterate through the images an processes all image sets
    log(f"[KaistToYolo::KaistToYolo] Kaist To Yolo formatting in '{dataset_format}' format:")
    for kaist_sets_path in kaist_sets_paths:
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
                new_dataset_label_paths = (
                    os.path.join(kaist_yolo_dataset_path,data_set_name,lwir_folder_name,labels_folder_name),
                    os.path.join(kaist_yolo_dataset_path,data_set_name,visible_folder_name,labels_folder_name))
                new_dataset_kaist_images_paths = (
                    os.path.join(kaist_yolo_dataset_path,data_set_name,lwir_folder_name,images_folder_name),
                    os.path.join(kaist_yolo_dataset_path,data_set_name,visible_folder_name,images_folder_name))
                
                for folder in (new_dataset_label_paths + new_dataset_kaist_images_paths):
                    # log(folder)
                    Path(folder).mkdir(parents=True, exist_ok=True)

                # Process all lines in imageSet file to create labelling in the new folder structure
                with open(file_path, 'r') as file:
                    lines_list = [line.rstrip().replace("\n", "") for line in file]

                    images_list_create = [image for image in lines_list if image not in pre_processed[visible_folder_name]]
                    images_list_symlink = [image for image in lines_list if image in pre_processed[visible_folder_name]]

                    with Pool() as pool:
                        func = partial(processLineImages, data_set_name, rgb_eq ,thermal_eq)
                        results = pool.map(func, images_list_create)

                        for result in results:
                            pre_processed[lwir_folder_name].update(result[lwir_folder_name])
                            pre_processed[visible_folder_name].update(result[visible_folder_name])

                    with Pool() as pool:
                        func = partial(processLineLabels, new_dataset_label_paths, dataset_format)
                        pool.map(func, images_list_create)

                    with Pool() as pool:
                        func = partial(upateProcessedSymlinks, pre_processed, data_set_name)
                        pool.map(func, images_list_symlink)
                        

                    log(f"\t· [{dataset_processed}] Processed {data_set_name} dataset: {len(lines_list)} XML files (and x2 images: visible and lwir) ({len(images_list_symlink)} as symlink).")
                dataset_processed += 1
                        
    # checkImageLabelPairs(kaist_yolo_dataset_path)
    log(f"[KaistToYolo::KaistToYolo] Finished procesing {dataset_processed} datasets. Output datasests are located in {kaist_yolo_dataset_path}")
    log(f"[KaistToYolo::KaistToYolo] Class dict is now: {class_data[dataset_format]}")

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