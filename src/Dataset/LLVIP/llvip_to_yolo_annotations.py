#!/usr/bin/env python3
# encoding: utf-8

"""
    llvip to Yolo annotation formatting: one *.txt file per image (if no objects in image, no *.txt file is required). 
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
import numpy as np

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    sys.path.append('./src/Dataset')

from utils import updateSymlink
from Dataset.constants import class_data, dataset_whitelist, dataset_blacklist, llvip_annotation_path, llvip_images_path, llvip_yolo_dataset_path, llvip_sets_paths
from Dataset.constants import images_folder_name, labels_folder_name, visible_folder_name, lwir_folder_name
from Dataset.th_equalization import th_equalization, rgb_equalization
# from .check_dataset import checkImageLabelPairs

from utils import log, bcolors


def check_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        log(f"Error with file {file_path}: {e}", bcolors.ERROR)
        return False

def check_image(file_path):
    try:
        img = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            print(f"[ERROR] Error with image {file_path}. Img is None or size is 0")
            return False
        
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            if np.all(img == 0) or np.all(img == 255):
                print(f"[ERROR] Error with image {file_path}. All 0 or 255.")
                return False
        else:
            if np.all(img == [0,0,0]) or np.all(img == [255,255,255]):
                print(f"[ERROR] Error with image {file_path}. All 0 or 255 in all channels.")
                return False
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            cv.cvtColor(img, cv.COLOR_BGR2RGB)
            cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        return True
    except Exception as e:
        log(f"Error with file {file_path}: {e}", bcolors.ERROR)
        return False

def processXML(xml_path, output_paths, dataset_format, relabeling):
    global class_data
    global obj_class_dict

    obj_class_dict = class_data[dataset_format]
    
    img_labels = []
    with open(xml_path) as xml:
        doc = untangle.parse(xml)
        if hasattr(doc.annotation, "object"):
            obj_name_list = [object.name.cdata for object in doc.annotation.object]
            for index,object in enumerate(doc.annotation.object):
                label = None
                obj_name = object.name.cdata #.replace("?","")

                if obj_name == "person?":
                    continue

                img_width = float(doc.annotation.size.width.cdata)
                img_height = float(doc.annotation.size.height.cdata)
                
                xmax = float(object.bndbox.xmax.cdata)
                ymax = float(object.bndbox.ymax.cdata)
                xmin = float(object.bndbox.xmin.cdata)
                ymin = float(object.bndbox.ymin.cdata)

                w,h = (xmax-xmin),(ymax-ymin)
                x,y = xmin+w/2, ymin+h/2

                if dataset_format == 'llvip_coco':
                    if obj_name == "people" or obj_name == "cyclist":
                        obj_name = "person"
                        
                    # Only processes person for now 
                    if obj_name == "person":
                        label = [obj_class_dict[obj_name], x,y,w,h]

                # For now llvip format takes only into account persons
                # Assumes llvip regular format
                # (dataset_format == 'llvip_small' or dataset_format == 'llvip_full')
                else:
                    # For now tag all classes
                    if obj_name == "cyclist":
                        obj_name = "person"
                    
                    if obj_name == "person":
                        label = [obj_class_dict[obj_name], x,y,w,h]
                    # label = [obj_class_dict[obj_name], x,y,w,h]
                
                img_labels.append(label)

            txt_data = ""
            for label in img_labels:
                obj, x_centered, y_centered, w, h = label
                x_normalized = x_centered / img_width
                y_normalized = y_centered / img_height
                w_normalized = float(w) / img_width
                h_normalized = float(h) / img_height
                txt_data += f"{obj} {x_normalized} {y_normalized} {w_normalized} {h_normalized}\n"

            # log(f"Processed {len(obj_name_list)} objects in image ({obj_name_list}); stored {len(img_labels)} labels in:"+\
            #     f"\n\t\t {output_paths}")
            # for file in output_paths:
            if len(output_paths) >=3:
                log(f"len(output_paths) >=3 - {len(output_paths) = }: {output_paths}", bcolors.ERROR)
            
            # First file made, second is symlink to first one
            if txt_data != "":
                with open(output_paths[0], 'w+') as output:
                    output.write(txt_data)
                
                updateSymlink(output_paths[0], output_paths[1])

                check_txt(output_paths[0])
                check_txt(output_paths[1])

# Process line from dataset file so to paralelice process
## IMPORTANT -> line has to be the last argument
def processLineLabels(new_dataset_label_paths, dataset_format, relabeling, line):
    file_name = line.split("/")[-1]
    
    root_label_path = os.path.join(llvip_annotation_path,f"{file_name}.xml")
    output_paths = [os.path.join(folder,f"{file_name}.txt") for folder in  new_dataset_label_paths]
    # log(output_paths)
    processXML(root_label_path, output_paths, dataset_format, relabeling)


def processLineImages(data_set_name, rgb_eq, thermal_eq, relabeling, line):
    processed = {lwir_folder_name: {}, visible_folder_name: {}}

    for data_type in (lwir_folder_name, visible_folder_name):
    
        # Create images
        image_path = os.path.join(data_type, line)
        root_image_path = os.path.join(llvip_images_path,f"{image_path}.jpg")
        file_name = line.split("/")[-1]
        new_image_path = os.path.join(llvip_yolo_dataset_path,data_set_name,data_type,images_folder_name,f"{file_name}.jpg")

        # print(f"{root_image_path = }; {new_image_path = }")
        # Apply clahe equalization to LWIR images if needed, or add symlink instead

        if 'lwir' in data_type:
            img = cv.imread(root_image_path, cv.IMREAD_GRAYSCALE)
        else:
            img = cv.imread(root_image_path)

        if 'lwir' in data_type and str(thermal_eq).lower() != 'none':
            img = th_equalization(img, thermal_eq)
            cv.imwrite(new_image_path, img)
        elif 'visible' in data_type and str(rgb_eq).lower() != 'none':
            # log(new_image_path)
            # Create or update symlink if already exists
            # updateSymlink(root_image_path, new_image_path)
            img = rgb_equalization(img, rgb_eq)
            cv.imwrite(new_image_path, img)
        else:
            # Create or update symlink if already exists
            # updateSymlink(root_image_path, new_image_path)
            ## Now needs transform anyway
            cv.imwrite(new_image_path, img)
        
        check_image(new_image_path)
        processed[data_type][line] = new_image_path
    
    return processed
    # log(f"[llvipToYolo::processLine] Process {root_label_path}")


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
    
# Distortion correct added just for compatibility
def llvipToYolo(dataset_format = 'llvip_coco', rgb_eq = 'none', thermal_eq = 'none', distortion_correct = None, relabeling = True):
    global class_data

    dataset_processed = 0
    pre_processed = {lwir_folder_name: {}, visible_folder_name: {}}
    # Goes to imageSets folder an iterate through the images an processes all image sets
    log(f"[llvipToYolo::llvipToYolo] llvip To Yolo formatting in '{dataset_format}' format:")
    for llvip_sets_path in llvip_sets_paths:
        for file in os.listdir(llvip_sets_path):
            file_path = os.path.join(llvip_sets_path, file)
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
                    os.path.join(llvip_yolo_dataset_path,data_set_name,lwir_folder_name,labels_folder_name),
                    os.path.join(llvip_yolo_dataset_path,data_set_name,visible_folder_name,labels_folder_name))
                new_dataset_llvip_images_paths = (
                    os.path.join(llvip_yolo_dataset_path,data_set_name,lwir_folder_name,images_folder_name),
                    os.path.join(llvip_yolo_dataset_path,data_set_name,visible_folder_name,images_folder_name))
                
                for folder in (new_dataset_label_paths + new_dataset_llvip_images_paths):
                    # log(folder)
                    Path(folder).mkdir(parents=True, exist_ok=True)

                # Process all lines in imageSet file to create labelling in the new folder structure
                with open(file_path, 'r') as file:
                    lines_list = [line.rstrip().replace("\n", "") for line in file]

                    images_list_create = [image for image in lines_list if image not in pre_processed[visible_folder_name]]
                    images_list_symlink = [image for image in lines_list if image in pre_processed[visible_folder_name]]

                    with Pool() as pool:
                        func = partial(processLineImages, data_set_name, rgb_eq, thermal_eq, relabeling)
                        results = pool.map(func, images_list_create)

                        for result in results:
                            pre_processed[lwir_folder_name].update(result[lwir_folder_name])
                            pre_processed[visible_folder_name].update(result[visible_folder_name])

                    with Pool() as pool:
                        func = partial(upateProcessedSymlinks, pre_processed, data_set_name)
                        pool.map(func, images_list_symlink)

                    with Pool() as pool:
                        func = partial(processLineLabels, new_dataset_label_paths, dataset_format, relabeling)
                        pool.map(func, images_list_create)

                    
                    log(f"\t· [{dataset_processed}] Processed {data_set_name} dataset: {len(lines_list)} XML files (and x2 images: visible and lwir) ({len(images_list_symlink)} as symlink).")
                dataset_processed += 1
                        
    # checkImageLabelPairs(llvip_yolo_dataset_path)
    log(f"[llvipToYolo::llvipToYolo] Finished procesing {dataset_processed} datasets. Output datasests are located in {llvip_yolo_dataset_path}")
    log(f"[llvipToYolo::llvipToYolo] Class dict is now: {class_data[dataset_format]}")


if __name__ == '__main__':
    llvipToYolo('llvip_80_20')
