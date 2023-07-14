#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates different approachs of mixing RGB with Thermal images
    hsvt -> combines v + t and reescalate that channel
    rgbt -> averages each chanel with thermal data (r+t/2)
"""

import os
from pathlib import Path
import shutil

from multiprocessing.pool import Pool
from functools import partial

import numpy as np
import cv2 

from config_utils import yolo_dataset_path, log

lwir = "/lwir/"
visible = "/visible/"

label_folder = "/labels/"
images_folder = "/images/"

# non_stop = True  # Set to false to visualize each image generated

def combine_hsvt(visible_image, thermal_image):
    h,s,v = cv2.split(cv2.cvtColor(visible_image, cv2.COLOR_BGR2HSV))
    th_channel = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)


    # Cast to 32S to avoid saturation when both channels are added
    v = v.astype(np.float64)
    th_channel = th_channel.astype(np.float64)

    intensity = v + th_channel
    _, max_val, _, _ = cv2.minMaxLoc(intensity)
    intensity = 255 * (intensity / max_val)
    intensity = intensity.astype(np.uint8)

    hsvt_image = cv2.merge([h, s, intensity])
    hsvt_image = cv2.cvtColor(hsvt_image, cv2.COLOR_HSV2BGR)

    return hsvt_image
                
def combine_rgbt(visible_image, thermal_image):
    b,g,r = cv2.split(visible_image)
    th_channel = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
    th_channel = th_channel.astype(np.float64)
    
    for ch in (b,g,r):
        ch = ch.astype(np.float64)
        ch = (ch + th_channel) / 2
        ch = ch.astype(np.uint8)

    rgbt_image = cv2.merge([b,g,r])
    
    return rgbt_image

def process_image(folder, combine_method, option_path, image):
    # global non_stop
    # log(f"Processing image {image} from {folder} dataset")

    thermal_image_path = f"{yolo_dataset_path}/{folder}/{lwir}/{images_folder}/{image}"
    rgb_image_path = f"{yolo_dataset_path}/{folder}/{visible}/{images_folder}/{image}"

    rgb_img = cv2.imread(rgb_image_path)
    th_img = cv2.imread(thermal_image_path) # It is enconded as BGR so still needs merging to Gray

    image_combined = combine_method(rgb_img, th_img)
    
    # log(f"Store new image into {option_path}/{image}")
    cv2.imwrite(f"{option_path}/{image}", image_combined)

    # if not non_stop:
    #     cv2.imshow(f"image {option}", image_combined)            
    
    # key = cv2.pollKey()
    # if key == ord('q') or key == ord('Q') or key == 27:
    #     cv2.destroyAllWindows()
    #     exit()
    # elif key == ord('c') or key == ord('C'):
    #     cv2.destroyAllWindows()
    #     non_stop = True

def make_dataset(folder, option, combine_method):

    # Images as new dataset option to new path with its labels
    option_path = f"{yolo_dataset_path}/{folder}/{option}/{images_folder}/".replace("//", "/")
    Path(option_path).mkdir(parents=True, exist_ok=True)
    shutil.copytree(f"{yolo_dataset_path}/{folder}/{lwir}/{label_folder}", 
                    f"{yolo_dataset_path}/{folder}/{option}/{label_folder}", 
                    dirs_exist_ok=True)

    log(f"Process {folder} dataset, output images will be stored in {option_path}")

    # Iterate images multiprocessing
    with Pool() as pool:
        images_list = os.listdir(f"{yolo_dataset_path}/{folder}/{lwir}/{images_folder}")
        func = partial(process_image, folder, combine_method, option_path)
        pool.map(func, images_list)
        
if __name__ == '__main__':
    # Iterate each of the datasets
    for folder in os.listdir(yolo_dataset_path):
        if not os.path.isdir(f"{yolo_dataset_path}/{folder}"):
            continue
        
        # make_dataset(folder, 'hsvt', combine_hsvt)
        make_dataset(folder, 'rgbt', combine_rgbt)
