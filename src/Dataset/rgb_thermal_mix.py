#!/usr/bin/env python3
# encoding: utf-8


import os, errno
from pathlib import Path
import shutil

from multiprocessing.pool import Pool
from functools import partial

import numpy as np
import cv2 as cv

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')

from utils import log, bcolors
# from .check_dataset import checkImageLabelPairs
from .constants import dataset_options, kaist_yolo_dataset_path, images_folder_name, labels_folder_name ,lwir_folder_name, visible_folder_name


test = None
test_plot = False

def process_image(folder, combine_method, option_path, image):
    # log(f"Processing image {image} from {folder} dataset")

    thermal_image_path = os.path.join(kaist_yolo_dataset_path,folder,lwir_folder_name,images_folder_name,image)
    rgb_image_path = os.path.join(kaist_yolo_dataset_path,folder,visible_folder_name,images_folder_name,image)

    rgb_img = cv.imread(rgb_image_path)
    th_img = cv.imread(thermal_image_path) # It is enconded as BGR so still needs merging to Gray

    image_combined = combine_method(rgb_img, th_img, path = f"{option_path}/{image}")
    # return image_combined

def make_dataset(option):
    if option not in dataset_options:
        log(f"[RGBThermalMix::make_dataset] Option {option} not found in dataset generation options. Not generating.", bcolors.WARNING)
        return
    
    symlink_created = 0
    processed_images = {}
    dataset_processed = 0
    # Iterate each of the datasets
    log(f"[RGBThermalMix::make_dataset] Process {option} option dataset:")
    for folder in os.listdir(kaist_yolo_dataset_path):
        if not os.path.isdir(os.path.join(kaist_yolo_dataset_path,folder)):
             continue
        
        # Images as new dataset option to new path with its labels
        option_path = os.path.join(kaist_yolo_dataset_path,folder,option,images_folder_name)
        Path(option_path).mkdir(parents=True, exist_ok=True)
        shutil.copytree(os.path.join(kaist_yolo_dataset_path,folder,lwir_folder_name,labels_folder_name), 
                        os.path.join(kaist_yolo_dataset_path,folder,option,labels_folder_name), 
                        dirs_exist_ok=True)

        images_list = os.listdir(os.path.join(kaist_yolo_dataset_path,folder,lwir_folder_name,images_folder_name))
        images_list_create = [image for image in images_list if image not in processed_images]
        images_list_symlink = [image for image in images_list if image in processed_images]
        
        # Only N images to be faster during testing. They are displayed and computed one by one
        if test:
            images_list_create = images_list_create[:test]
            for image in images_list_create[:test]:
                # Creating visualization windows
                process_image(folder, dataset_options[option]['merge'], option_path, image)
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
                func = partial(process_image, folder, dataset_options[option]['merge'], option_path)
                pool.map(func, images_list_create)
        
        # Symlink
        for image in images_list_symlink:
            symlink_created +=1
            current_image = processed_images[image].replace('.png', dataset_options[option]['extension'])
            img_path = os.path.join(option_path,image).replace('.png', dataset_options[option]['extension'])
            try:
                os.symlink(current_image, img_path)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(img_path)
                    os.symlink(current_image, img_path)
                else:
                    raise e
        log(f"\t· [{dataset_processed}] Processed {folder} dataset ({len(images_list_create)} images; {len(images_list_symlink)} symlink), output images were stored in {option_path}")

        dataset_processed += 1
        processed_images = {**processed_images, **{image: os.path.join(option_path,image) for image in images_list_create}}
        # log(f"Not creating images as they already exist, creating symlink to previous generated image: {images_list_symlink}")
        
        if test:
            log(f"Test mode enabled for {test} images. Finished processing {folder}.")
            break

        # checkImageLabelPairs(os.path.join(kaist_yolo_dataset_path,folder,option))
    log(f"[RGBThermalMix::make_dataset] Created {symlink_created} symlinks instead of repeating images.")

if __name__ == '__main__':
    
    test = 500
    test_plot = False

    from argparse import ArgumentParser

    option_list_default = dataset_options.keys()
    arg_dict = {}
    parser = ArgumentParser(description="Dataset generation with fussed images between visual and thermal.")
    parser.add_argument('-o', '--option', action='store', dest='olist', metavar='OPTION',
                        type=str, nargs='*', default=option_list_default,
                        help=f"Option of the dataset to be used. Available options are {option_list_default}. Usage: -c item1 item2, -c item3")
    
    opts = parser.parse_args()

    dataset_generate = dataset_options.keys() # list(opts.olist)
    log(f"Compute datasets for {dataset_generate} conditions.")
    
    if test:
        log(f"Only computes subtest of {test} images for each dataset as test mode is enabled.")
    
    for option in dataset_generate:
        make_dataset(option)