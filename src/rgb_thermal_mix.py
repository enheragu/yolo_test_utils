#!/usr/bin/env python3
# encoding: utf-8

"""
    Creates different approachs of mixing RGB with Thermal images
"""

import os
from pathlib import Path
import shutil

import numpy as np
import cv2 

from config import yolo_dataset_path, log

lwir = "/lwir/"
visible = "/visible/"
hsvt = "/hsvt/"

label_folder = "/labels/"
images_folder = "/images/"


non_stop = False
if __name__ == '__main__':
    # Iterate each of the datasets
    for folder in os.listdir(yolo_dataset_path):
        if not os.path.isdir(f"{yolo_dataset_path}/{folder}"):
            continue
        
        # Images as hsvt to new path with its labels
        hsvt_path = f"{yolo_dataset_path}/{folder}/{hsvt}/{images_folder}/".replace("//", "/")
        Path(hsvt_path).mkdir(parents=True, exist_ok=True)
        shutil.copytree(f"{yolo_dataset_path}/{folder}/{lwir}/{label_folder}", 
                        f"{yolo_dataset_path}/{folder}/{hsvt}/{label_folder}", 
                        dirs_exist_ok=True)

        log(f"Process {folder} dataset, output images will be stored in {hsvt_path}")

        # Iterate images
        for image in os.listdir(f"{yolo_dataset_path}/{folder}/{lwir}/{images_folder}"):
            # log(f"Processing image {image} from {folder} dataset")

            thermal_image_path = f"{yolo_dataset_path}/{folder}/{lwir}/{images_folder}/{image}"
            rgb_image_path = f"{yolo_dataset_path}/{folder}/{visible}/{images_folder}/{image}"

            rgb_img = cv2.imread(rgb_image_path)
            th_img = cv2.imread(thermal_image_path) # It is enconded as BGR so still needs merging to Gray

            h,s,v = cv2.split(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV))
            th_channel = cv2.cvtColor(th_img, cv2.COLOR_BGR2GRAY)


            # Cast to 32S to avoid saturation when both channels are added
            v = v.astype(np.float64)
            th_channel = th_channel.astype(np.float64)

            intensity = v + th_channel
            _, max_val, _, _ = cv2.minMaxLoc(intensity)
            intensity = 255 * (intensity / max_val)
            intensity = intensity.astype(np.uint8)


            hsvt_image = cv2.merge([h, s, intensity])
            hsvt_image = cv2.cvtColor(hsvt_image, cv2.COLOR_HSV2BGR)
            cv2.imwrite(f"{hsvt_path}/{image}", hsvt_image)
                       
            # log(f"Stored new image into mage {hsvt_path}/{image}; shape: {hsvt_image.shape}")

            if not non_stop:
                cv2.imshow("image hsv", hsvt_image)            
            
            key = cv2.pollKey()
            if key == ord('q') or key == ord('Q') or key == 27:
                cv2.destroyAllWindows()
                exit()
            elif key == ord('c') or key == ord('C'):
                cv2.destroyAllWindows()
                non_stop = True