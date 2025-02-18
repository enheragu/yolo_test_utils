#!/usr/bin/env python3
# encoding: utf-8


import os, errno
from pathlib import Path
import shutil
import time

from statistics import mean, stdev

from multiprocessing.pool import Pool
from functools import partial

import numpy as np
import cv2 as cv

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')

from utils import log, bcolors, updateSymlink
# from .check_dataset import checkImageLabelPairs
from Dataset.constants import dataset_options, kaist_yolo_dataset_path, images_folder_name, labels_folder_name ,lwir_folder_name, visible_folder_name
from Dataset.th_equalization import th_equalization, rgb_equalization


test = None
test_plot = False

def process_image(yolo_dataset_path, folder, combine_method, option_path, dataset_format, rgb_eq, thermal_eq, image):
    # log(f"Processing image {image} from {folder} dataset")

    thermal_image_path = os.path.join(yolo_dataset_path,folder,lwir_folder_name,images_folder_name,image)
    rgb_image_path = os.path.join(yolo_dataset_path,folder,visible_folder_name,images_folder_name,image)

    rgb_img = cv.imread(rgb_image_path)
    th_img = cv.imread(thermal_image_path, cv.IMREAD_GRAYSCALE) # It is enconded as BGR so still needs merging to Gray


    start_time = time.perf_counter()
    
    th_img = th_equalization(th_img, thermal_eq)
    rgb_img = rgb_equalization(rgb_img, rgb_eq)

    image_combined = combine_method(rgb_img, th_img, path = f"{option_path}/{image}")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    return image_combined, total_time

def make_dataset(option, dataset_format = 'kaist_coco', rgb_eq = 'none', thermal_eq = 'none', yolo_version_dataset_path = kaist_yolo_dataset_path):
    if option not in dataset_options:
        log(f"[RGBThermalMix::make_dataset] Option {option} not found in dataset generation options. Not generating.", bcolors.WARNING)
        return
    
    symlink_created = 0
    processed_images = {}
    dataset_processed = 0
    execution_time_all = []
    # Iterate each of the datasets
    log(f"[RGBThermalMix::make_dataset] Process {option} option dataset:")
    for folder in os.listdir(yolo_version_dataset_path):
        if not os.path.isdir(os.path.join(yolo_version_dataset_path,folder)):
             continue
        
        # Images as new dataset option to new path with its labels
        option_path = os.path.join(yolo_version_dataset_path,folder,option,images_folder_name)
        Path(option_path).mkdir(parents=True, exist_ok=True)
        # Symlink to labels instead of deepcopy to save memory
        updateSymlink(os.path.join(yolo_version_dataset_path,folder,lwir_folder_name,labels_folder_name), 
                        os.path.join(yolo_version_dataset_path,folder,option,labels_folder_name))
        # shutil.copytree(os.path.join(yolo_version_dataset_path,folder,lwir_folder_name,labels_folder_name), 
        #                 os.path.join(yolo_version_dataset_path,folder,option,labels_folder_name), 
        #                 dirs_exist_ok=True)

        images_list = os.listdir(os.path.join(yolo_version_dataset_path,folder,lwir_folder_name,images_folder_name))
        images_list_create = [image for image in images_list if image not in processed_images]
        images_list_symlink = [image for image in images_list if image in processed_images]
        
        # Only N images to be faster during testing. They are displayed and computed one by one
        if test:
            images_list_create = images_list_create[:test]
            for image in images_list_create[:test]:
                # Creating visualization windows
                process_image(yolo_version_dataset_path, folder, dataset_options[option]['merge'], option_path, dataset_format, rgb_eq, thermal_eq, image)
                if test_plot:
                    fused_image = cv.imread(option_path + image)
                    cv.namedWindow("Image fussion", cv.WINDOW_AUTOSIZE)
                    cv.imshow("Image fussion", fused_image)     # Display the resulting frame

                    # check keystroke to exit (image window must be on focus)
                    key = cv.pollKey()
                    # key = cv.waitKey()
                    if key == ord('q') or key == ord('Q') or key == 27:
                        break
        else:
            # Iterate images multiprocessing
            # with Pool(processes = 5) as pool:
            with Pool() as pool:    
                func = partial(process_image, yolo_version_dataset_path, folder, dataset_options[option]['merge'], option_path, dataset_format, rgb_eq, thermal_eq)
                results = pool.map(func, images_list_create)

                execution_times = [result[1] for result in results]
                execution_time_all.extend(execution_times)
        
        # Symlink
        for image in images_list_symlink:
            symlink_created +=1
            current_image = processed_images[image].replace('.png', dataset_options[option]['extension'])
            img_path = os.path.join(option_path,image).replace('.png', dataset_options[option]['extension'])
            updateSymlink(current_image, img_path)
        log(f"\t· [{dataset_processed}] Processed {folder} dataset ({len(images_list_create)} images; {len(images_list_symlink)} symlink), output images were stored in {option_path}")

        dataset_processed += 1
        processed_images = {**processed_images, **{image: os.path.join(option_path,image) for image in images_list_create}}
        # log(f"Not creating images as they already exist, creating symlink to previous generated image: {images_list_symlink}")
        
        if test:
            log(f"Test mode enabled for {test} images. Finished processing {folder}.")
            break

        # checkImageLabelPairs(os.path.join(yolo_version_dataset_path,folder,option))
    log(f"[RGBThermalMix::make_dataset] Created {symlink_created} symlinks instead of repeating images.")
    log(f"[RGBThermalMix::make_dataset] Fussion method for option {option} took on average {mean(execution_time_all)}s (std: {stdev(execution_time_all)}) on each image.")


if __name__ == '__main__':
    
    test = None
    test_plot = False

    from argparse import ArgumentParser

    option_list_default = dataset_options.keys()
    arg_dict = {}
    parser = ArgumentParser(description="Dataset generation with fussed images between visual and thermal.")
    parser.add_argument('-o', '--option', action='store', dest='olist', metavar='OPTION',
                        type=str, nargs='*', default=option_list_default,
                        help=f"Option of the dataset to be used. Available options are {option_list_default}. Usage: -o item1 item2, -o item3")
    
    opts = parser.parse_args()

    dataset_generate = list(opts.olist)  # dataset_options.keys()
    log(f"Compute datasets for {dataset_generate} conditions.")
    
    if test:
        log(f"Only computes subtest of {test} images for each dataset as test mode is enabled.")
    
    for option in dataset_generate:
        make_dataset(option)