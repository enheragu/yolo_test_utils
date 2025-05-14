
#!/usr/bin/env python3
# encoding: utf-8

"""
    Script to evaluate dataset image distribution and handle the generation of new
    set distribution for test/train
"""

import os  
from pathlib import Path
from itertools import zip_longest

if __name__ == "__main__":
    import sys
    sys.path.append('./src')
from utils.log_utils import logTable

home = Path.home()

llvip_dataset_path = f"{home}/eeha/LLVIP"
llvip_yolo_dataset_path = f"{home}/eeha/llvip-yolo-annotated/" # Output dataset in YOLO format

labels_folder_name = "labels"
images_folder_name = "images"
visible_folder_name = "visible"
lwir_folder_name = "lwir"

store_path = f"{home}/eeha/dataset_analysis/llvip"
 

# Function to count images in a directory
def count_images(path, unique_images = None):
    total = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".npy", ".npz")):
                total += 1
                if unique_images != None:
                    unique_images.add(file)
    return total

# Function to count lines in text files in a directory
def count_lines(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                labels_path = os.path.join(root, file)
                if os.path.exists(labels_path):
                    with open(labels_path, "r") as f:
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


def evaluateOutputDataset(white_list = [], title = "dataset_info.txt"):
    
    # Dictionary to store the count of images and lines of output dataset
    count = {"night": {"test": {"images": 0, "backgrounds": 0, "b_percent": 0, "instances": 0, "unique_images": set()},
                       "train": {"images": 0, "backgrounds": 0, "b_percent": 0, "instances": 0, "unique_images": set()}}
            }

    evaluated_dirs = []
    # Iterate over the directories and count images and lines
    # for root, dirs, files in os.walk(llvip_yolo_dataset_path):
    #     for directory in dirs:
    root = llvip_yolo_dataset_path
    for directory in os.listdir(llvip_yolo_dataset_path):
        directory_path = os.path.join(root, directory)
        if os.path.isdir(directory_path):
            if directory.startswith(("test", "train")):               
                if white_list == [] or directory in white_list:
                    evaluated_dirs.append(directory)
                    parts = directory.split("-")
                    print(f"{directory =}")
                    if len(parts) == 3:
                        type, condition, id = parts
                        visible_path = os.path.join(root, directory, visible_folder_name)
                        count[condition][type]["images"] += count_images(os.path.join(visible_path, images_folder_name), count[condition][type]["unique_images"])
                        count[condition][type]["instances"] += count_lines(os.path.join(visible_path, labels_folder_name))
                        count[condition][type]["backgrounds"] += count_backgrounds(visible_path)
                    else:
                        print(f"Ignoring directory '{directory}' as it doesn't match the expected format.")
                else:
                    pass
                    # print(f"Directory {directory} not in whitelist: {white_list}")

    # Print results
    print(f"Evaluated dirs: {evaluated_dirs}")

    for condition in ["night"]:
        set_unique = set()
        total_len = 0
        for type in ["test", "train"]:
            n_images = count[condition][type]['images']
            if n_images == 0:
                del count[condition][type]
                continue
            count[condition][type]["b_percent"] = round(count[condition][type]['backgrounds']*100.0/n_images, 3)
            set_unique.update(count[condition][type]['unique_images'])
            total_len += len(count[condition][type]['unique_images'])
            count[condition][type]["unique_images"] = len(set_unique)/total_len if total_len != 0 else total_len

    headers = [""] + list(count['night']['test'].keys())
    # row_header = [f"{key} {sub_key}" for key in count.keys() for sub_key in count[key].keys()]

    # Crear una lista de listas para los datos de la tabla
    table_data = [headers]
    for key, value in count.items():
        for sub_key, sub_value in value.items():
            row = [f"{sub_key} {key}"] + [sub_value[key] for key in sub_value.keys()]
            table_data.append(row)


    file_log_path = os.path.join(store_path, 'dataset_info')
    if not os.path.exists(file_log_path):
        os.makedirs(file_log_path)

    logTable(table_data, file_log_path, title.replace('.txt',''))

    return count



set_info = {'night': {'num_img': 0, 'test': [], 'train': []}}
def evaluateInputDataset(title = 'input_dataset.txt'):
    print()
    log_table = str()
    for folder_set in os.listdir(llvip_dataset_path):
        set_nun_img = 0
        for subfolder_set in os.listdir(os.path.join(llvip_dataset_path, folder_set)):
            path_set = os.path.join(llvip_dataset_path, folder_set, subfolder_set)
            if os.path.isdir(path_set):
                visible_folder = os.path.join(path_set, visible_folder_name)
                if os.path.exists(visible_folder):
                    nun_img = count_images(visible_folder)
                    set_nun_img += nun_img
                    # print(f"Set: {folder_set}/{subfolder_set} - Num Imags: {nun_img}")
        
        for condition in set_info.values():
            if folder_set in condition['sets']:
                condition['num_img'] += set_nun_img
        log_table += f"Set: {folder_set} - Num Imags: {set_nun_img}\n"
        print(f"Set: {folder_set} - Num Imags: {set_nun_img}")

    log_table += f"\nTotal night: {set_info['night']['num_img']}\n"
    print(f"\nTotal night: {set_info['night']['num_img']}\n")

    file_log_path = os.path.join(store_path, 'dataset_info')
    if not os.path.exists(file_log_path):
        os.makedirs(file_log_path)

    file_log = os.path.join(file_log_path, title)
    with open(file_log, 'w') as file:
        file.write(log_table)

    # print(f"Split img   -> (train - test)")
    # print(f"Day   70-30 -> {int(set_info['day']['num_img']*0.7)} - {int(set_info['day']['num_img']*0.3)}")
    # print(f"Night 70-30 -> {int(set_info['night']['num_img']*0.7)} - {int(set_info['night']['num_img']*0.3)}")
    # print(f"Day   80-20 -> {int(set_info['day']['num_img']*0.8)} - {int(set_info['day']['num_img']*0.2)}")
    # print(f"Night 80-20 -> {int(set_info['night']['num_img']*0.8)} - {int(set_info['night']['num_img']*0.2)}")
    # print(f"Day   90-10 -> {int(set_info['day']['num_img']*0.9)} - {int(set_info['day']['num_img']*0.1)}")
    # print(f"Night 90-10 -> {int(set_info['night']['num_img']*0.9)} - {int(set_info['night']['num_img']*0.1)}")

if __name__ == '__main__':

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # evaluateOutputDataset()
    # evaluateInputDataset()
    # splitDatsets()

    
    # evaluateOutputDataset(white_list=['test-all-01','train-all-01','test-day-01','train-day-02','test-night-01','train-night-02'], title="dataset_kaist.txt")
    # print("\nCase 70-30:")
    # evaluateOutputDataset(white_list=['test-day-70_30','train-day-70_30','test-night-70_30','train-night-70_30'], title="dataset_70-30.txt")
    print("\nCase 80-20:")
    evaluateOutputDataset(white_list=['test-night-80_20','train-night-80_20'], title="dataset_80-20.txt")
    # print("\nCase 90-10:")
    # evaluateOutputDataset(white_list=['test-day-90_10','train-day-90_10','test-night-90_10','train-night-90_10'], title="dataset_90-10.txt")