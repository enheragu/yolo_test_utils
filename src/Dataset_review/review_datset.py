
#!/usr/bin/env python3
# encoding: utf-8

import os  
from pathlib import Path

home = Path.home()
kaist_dataste_path = f"{home}/eeha/kaist-cvpr15/images"
kaist_yolo_dataset_path = f"{home}/eeha/kaist-yolo-annotated/"
labels_folder_name = "labels"
images_folder_name = "images"
visible_folder_name = "visible"

"""
    Script that evaluates how many images there are on each case along with how many
    instances are labeled
"""

import os

# Main directory where the directories of interest are located

# Dictionary to store the count of images and lines of output dataset
count = {
    "test": {"all": {"images": 0, "backgrounds": 0, "labels": 0, "unique_images": set()},
             "day": {"images": 0, "backgrounds": 0, "labels": 0, "unique_images": set()},
             "night": {"images": 0, "backgrounds": 0, "labels": 0, "unique_images": set()}},
    "train": {"all": {"images": 0, "backgrounds": 0, "labels": 0, "unique_images": set()},
              "day": {"images": 0, "backgrounds": 0, "labels": 0, "unique_images": set()},
              "night": {"images": 0, "backgrounds": 0, "labels": 0, "unique_images": set()}}
}
 
# Output dataset whitelist
white_list = ['test-all-01','train-all-01','test-day-01','train-day-02','test-night-01','train-night-02']

# Function to count images in a directory
def count_images(path, unique_images = None):
    total = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".npy", ".npz")):
                total += 1
                if unique_images:
                    unique_images.add(file)
    return total

# Function to count lines in text files in a directory
def count_lines(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r") as f:
                    total += sum(1 for _ in f)
    return total

def count_backgrounds(path):
    backgrounds = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".npy", ".npz")):
                image_name, _ = os.path.splitext(file)
                txt_file = os.path.join(root, "..", labels_folder_name, image_name + ".txt")
                if not os.path.isfile(txt_file) or os.path.getsize(txt_file) == 0:
                    backgrounds += 1
    return backgrounds


def evaluateOutputDataset():
    # Iterate over the directories and count images and lines
    for root, dirs, files in os.walk(kaist_yolo_dataset_path):
        for directory in dirs:
            if directory.startswith(("test", "train")):
                if directory in white_list:
                    parts = directory.split("-")
                    if len(parts) == 3:
                        type, condition, id = parts
                        visible_path = os.path.join(root, directory, visible_folder_name)
                        count[type][condition]["images"] += count_images(os.path.join(visible_path, images_folder_name), count[type][condition]["unique_images"])
                        count[type][condition]["labels"] += count_lines(os.path.join(visible_path, labels_folder_name))
                        count[type][condition]["backgrounds"] += count_backgrounds(visible_path)
                    else:
                        print(f"Ignoring directory '{directory}' as it doesn't match the expected format.")
                else:
                    print(f"Directory {directory} not in whitelist: {white_list}")


    # Print results
                
    print("\t\t& Images & Backgrounds & Instances & Unique Images \\\\")
    print("\t\t\\hline")

    log_str = ""
    for condition in ["all", "day", "night"]:
        set_unique = set()
        total_len = 0
        for type in ["Test", "Train"]:
            print(f"\t\t{type} {condition} & {count[type.lower()][condition]['images']} & {count[type.lower()][condition]['backgrounds']} & {count[type.lower()][condition]['labels']} & {len(count[type][condition]['unique_images'])} \\\\")
            set_unique.update(count[type.lower()][condition]['unique_images'])
            total_len +=len(count[type.lower()][condition]['unique_images'])
        log_str+=f"{condition}: Unique images between both train/test: {len(set_unique)}/{total_len}\n"
        print("\t\t\\hline")

    print(f"\n\n{log_str}")


### Train ###
# · Set 00 - Day / Campus / 5.92GB / 17,498 frames / 11,016 objects
# · Set 01 - Day / Road / 2.82GB / 8,035 frames / 8,550 objects
# · Set 02 - Day / Downtown / 3.08GB / 7,866 frames / 11,493 objects
# · Set 03 - Night / Campus / 2.40GB / 6,668 frames / 7,418 objects
# · Set 04 - Night / Road / 2.88GB / 7,200 frames / 17,579 objects
# · Set 05 - Night / Downtown / 1.01GB / 2,920 frames / 4,655 objects
    
### Test ###
# · Set 06 - Day / Campus / 4.78GB / 12,988 frames / 12,086 objects
# · Set 07 - Day / Road / 3.04GB / 8,141 frames / 4,225 objects
# · Set 08 - Day / Downtown / 3.50GB / 8,050 frames / 23,309 objects
# · Set 09 - Night / Campus / 1.38GB / 3,500 frames / 3,577 objects
# · Set 10 - Night / Road / 3.75GB / 8,902 frames / 4,987 objects
# · Set 11 - Night / Downtown / 1.33GB / 3,560 frames / 6,655 objects


def evaluateInputDataset():

    for folder_set in os.listdir(kaist_dataste_path):
        set_nun_img = 0
        for subfolder_set in os.listdir(os.path.join(kaist_dataste_path, folder_set)):
            path_set = os.path.join(kaist_dataste_path, folder_set, subfolder_set)
            if os.path.isdir(path_set):
                visible_folder = os.path.join(path_set, visible_folder_name)
                if os.path.exists(visible_folder):
                    nun_img = count_images(visible_folder)
                    set_nun_img += nun_img
                    # print(f"Set: {folder_set}/{subfolder_set} - Num Imags: {nun_img}")
        print(f"Set: {folder_set} - Num Imags: {set_nun_img}")


if __name__ == '__main__':
    # evaluateOutputDataset()
    evaluateInputDataset()