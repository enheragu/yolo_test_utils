#!/usr/bin/env python3
# encoding: utf-8
"""
    Kaist iterates over Kaist folder plotting label and detection over the images
    taking information about class and oclusion level from XML annotation files.
"""

import untangle
import yaml
import os, errno
from pathlib import Path
import shutil
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    sys.path.append('./src/Dataset')

from Dataset.constants import kaist_yolo_dataset_path
from utils.log_utils import log

labeled_images = "./kaist_yolo_labeled_images/"

lwir = "/lwir/"
visible = "/visible/"

class_color = {  'person': (114,196,83), 'person?': (70,133,46), 'cyclist': (26,209,226), 'people': (229,29,46) }
class_key = { 0: 'person', 1: 'cyclist', 2: 'people', 3: 'person?'}
""" 
    Class id with center and w,h of label in normalized coords
    0 0.9090909090909091 0.5443181818181818 0.03636363636363636 0.08863636363636364
"""
def procesLabelTXT(label_path, image_path):
    annotation = ()
    image = cv.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not open image {image_path}")
    
    with open(label_path) as file:
        components_list = []
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    components_list.append([float(part) for part in parts])
                except ValueError as e:
                    print(f"Error parsing line '{line}': {e}")
                    continue

        # log(f"File with {len(lines)} lines resulted in {len(components_list)} labels draw")    
        for components in components_list:        
            img_height, img_width, _ = image.shape
            cx = components[1] * img_width
            cy = components[2] * img_height
            w = components[3] * img_width 
            h = components[4] * img_height
            
            start_point = (int(cx - w/2), int(cy - h/2))
            end_point = (int(cx + w/2), int(cy + h/2))
            
            obj_name = class_key[components[0]]
            cv.rectangle(image, start_point, end_point, color=class_color[obj_name], thickness=1)

            label_str = f"{obj_name}"
            
            # For the text background
            # Finds space required by the text so that we can put a background with that amount of width.
            (w, h), _ = cv.getTextSize(label_str, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Prints the text.    
            image = cv.rectangle(image, (int(start_point[0]), int(start_point[1]-h-8)), (int(start_point[0]+w+4), int(start_point[1])), class_color[obj_name], -1)
            image = cv.putText(image, label_str, (int(start_point[0]+4), int(start_point[1]-h//2)),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
 
    return image

def processFile(file, subdir, test_path):
    # try:
        label_path = f"{subdir}/{file}"
        for img_type in ('/lwir/', '/visible/'):
            img_path = kaist_yolo_dataset_path + test_path.replace('/lwir/labels', img_type).replace('/visible/labels', img_type) + "/images/"
            output_image_path = img_path.replace(kaist_yolo_dataset_path, labeled_images).replace("/images/", "")

            Path(output_image_path).mkdir(parents=True, exist_ok=True)

            image_file = img_path + file.replace(".txt", ".png")
            
            if os.path.islink(image_file):
                link_target = os.readlink(image_file)
                labeled_image_link = link_target.replace(kaist_yolo_dataset_path, labeled_images).replace("/images/", "")
                if os.path.exists(output_image_path) and os.path.islink(output_image_path):
                    os.unlink(output_image_path)
                os.symlink(labeled_image_link, output_image_path)
                continue

            # print(f"Process:\n\t· IMG: {image_file}\n\t· XML: {label_path}\n\t· Output: {output_image_path}")
            image = procesLabelTXT(label_path, image_file)
            if image is None:
                # Empty labels
                continue
            # Resize image to save memory
            scale_percent = 60 # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
            # Save the resized image
            cv.imwrite(output_image_path + file.replace(".txt", ".png"), resized)

    # except Exception as e:
    #     print(f"Error processing file {label_path}: {e}")
        

if __name__ == '__main__':
    
    # Iterate XML folder to process it, gets image associated with the file and 
    # prepare output folder for iages
    print(f"Process {kaist_yolo_dataset_path} dataset")


    with ThreadPoolExecutor() as executor:
        futures = []
        for subdir, dirs, files in os.walk(kaist_yolo_dataset_path):    
            if len(files) == 0:
                continue
            
            test_path = subdir.replace(kaist_yolo_dataset_path, "")
            files_filtered = list(filter(lambda file: ".txt" in file, files))

            print(f"Process {test_path} dataset")

            for file in files_filtered:
                futures.append(executor.submit(processFile, file, subdir, test_path))

        for future in futures:
            future.result()
        
    cv.destroyAllWindows()