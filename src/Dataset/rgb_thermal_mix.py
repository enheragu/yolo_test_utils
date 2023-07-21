#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates different approachs of mixing RGB with Thermal images
    hsvt -> combines v + t and reescalate that channel
    rgbt -> averages each chanel with thermal data (r+t/2)
    4ch -> Stores ndarray with all foru channels as [b,g,r,t]
"""

import os, errno
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

def combine_hsvt(visible_image, thermal_image, path):
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
    
    cv2.imwrite(path, hsvt_image)
    return hsvt_image
      
              
def combine_rgbt(visible_image, thermal_image, path):
    b,g,r = cv2.split(visible_image)
    th_channel = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
    th_channel = th_channel.astype(np.float64)
    
    for ch in (b,g,r):
        ch = ch.astype(np.float64)
        ch = (ch + th_channel) / 2
        ch = ch.astype(np.uint8)

    rgbt_image = cv2.merge([b,g,r])
    
    cv2.imwrite(path, rgbt_image)
    return rgbt_image

def combine_4ch(visible_image, thermal_image, path):
    b,g,r = cv2.split(visible_image)
    th_channel = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)

    ch4_image = cv2.merge([b,g,r,th_channel])

    # cv2.imwrite(path, ch4_image)
    np.save(path.replace('.png',''), ch4_image)
    return ch4_image

def process_image(folder, combine_method, option_path, image):
    # global non_stop
    # log(f"Processing image {image} from {folder} dataset")

    thermal_image_path = f"{yolo_dataset_path}/{folder}/{lwir}/{images_folder}/{image}"
    rgb_image_path = f"{yolo_dataset_path}/{folder}/{visible}/{images_folder}/{image}"

    rgb_img = cv2.imread(rgb_image_path)
    th_img = cv2.imread(thermal_image_path) # It is enconded as BGR so still needs merging to Gray

    image_combined = combine_method(rgb_img, th_img, path = f"{option_path}/{image}")
    
    # if not non_stop:
    #     cv2.imshow(f"image {option}", image_combined)            
    
    # key = cv2.pollKey()
    # if key == ord('q') or key == ord('Q') or key == 27:
    #     cv2.destroyAllWindows()
    #     exit()
    # elif key == ord('c') or key == ord('C'):
    #     cv2.destroyAllWindows()
    #     non_stop = True




# Dict with tag-function to be used
dataset_options = {
                    'hsvt': combine_hsvt,
                    'rgbt': combine_rgbt,
                    '4ch': combine_4ch
                }


def make_dataset(option):
    symlink_created = 0
    processed_images = {}
    dataset_processed = 0
    # Iterate each of the datasets
    log(f"[RGBThermalMix::make_dataset] Process {option} option dataset:")
    for folder in os.listdir(yolo_dataset_path):
        if not os.path.isdir(f"{yolo_dataset_path}/{folder}"):
             continue
        
        # Images as new dataset option to new path with its labels
        option_path = f"{yolo_dataset_path}/{folder}/{option}/{images_folder}/".replace("//", "/")
        Path(option_path).mkdir(parents=True, exist_ok=True)
        shutil.copytree(f"{yolo_dataset_path}/{folder}/{lwir}/{label_folder}", 
                        f"{yolo_dataset_path}/{folder}/{option}/{label_folder}", 
                        dirs_exist_ok=True)

        images_list = os.listdir(f"{yolo_dataset_path}/{folder}/{lwir}/{images_folder}")
        images_list_create = [image for image in images_list if image not in processed_images]
        images_list_symlink = [image for image in images_list if image in processed_images]
        
        # Iterate images multiprocessing
        with Pool() as pool:
            func = partial(process_image, folder, dataset_options[option], option_path)
            pool.map(func, images_list_create)
        
        # Symlink
        for image in images_list_symlink:
            symlink_created +=1
            try:
                os.symlink(processed_images[image], f"{option_path}/{image}")
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(f"{option_path}/{image}")
                    os.symlink(processed_images[image], f"{option_path}/{image}")
                else:
                    raise e
        log(f"\t· [{dataset_processed}] Processed {folder} dataset ({len(images_list_create)} images; {len(images_list_symlink)} symlink), output images were stored in {option_path}")

        dataset_processed += 1
        processed_images = {**processed_images, **{image: f"{option_path}/{image}" for image in images_list_create}}
        # log(f"Not creating images as they already exist, creating symlink to previous generated image: {images_list_symlink}")
        
    log(f"[RGBThermalMix::make_dataset] Created {symlink_created} symlinks instead of repeating images.")

if __name__ == '__main__':
    for key, function in dataset_options.items():
        make_dataset(key)

    # make_dataset('hsvt', dataset_options['hsvt'])
    # make_dataset('rgbt', dataset_options['rgbt'])
    # make_dataset('4ch', dataset_options['4ch'])
