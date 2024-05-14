
#!/usr/bin/env python3
# encoding: utf-8

"""
    Script to evaluate dataset image distribution and handle the generation of new
    set distribution for test/train
"""

import os  
from pathlib import Path
from itertools import zip_longest
from tabulate import tabulate

home = Path.home()
kaist_dataset_path = f"{home}/eeha/kaist-cvpr15/images"
kaist_yolo_dataset_path = f"{home}/eeha/kaist-yolo-annotated/"
labels_folder_name = "labels"
images_folder_name = "images"
visible_folder_name = "visible"


 
# Output dataset whitelist (default is original Kaist sets)
white_list = ['test-all-01','test-day-20','train-all-01','train-all-20','train-day-20','train-night-20',
              'test-all-20','test-night-01','train-all-02','train-day-02','train-night-02',
              'test-day-01','test-night-20','train-all-04','train-day-04','train-night-04']

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


def evaluateOutputDataset(white_list = white_list, print_latex_table = True):
    
    # Dictionary to store the count of images and lines of output dataset
    count = {"all": {"test": {"images": 0, "backgrounds": 0, "b_percent": 0, "instances": 0, "unique_images": set()},
                     "train": {"images": 0, "backgrounds": 0, "b_percent": 0, "instances": 0, "unique_images": set()}},
             "day": {"test": {"images": 0, "backgrounds": 0, "b_percent": 0, "instances": 0, "unique_images": set()},
                     "train": {"images": 0, "backgrounds": 0, "b_percent": 0, "instances": 0, "unique_images": set()}},
             "night": {"test": {"images": 0, "backgrounds": 0, "b_percent": 0, "instances": 0, "unique_images": set()},
                       "train": {"images": 0, "backgrounds": 0, "b_percent": 0, "instances": 0, "unique_images": set()}}
            }

    evaluated_dirs = []
    # Iterate over the directories and count images and lines
    for root, dirs, files in os.walk(kaist_yolo_dataset_path):
        for directory in dirs:
            if directory.startswith(("test", "train")):
                if white_list == [] or directory in white_list:
                    evaluated_dirs.append(directory)
                    parts = directory.split("-")
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
    if print_latex_table:
        print("\t\t& Images & Backgrounds & Instances & Unique Images \\\\")
        print("\t\t\\hline")

    log_str = ""
    for condition in ["all", "day", "night"]:
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

            if print_latex_table:
                print(f"\t\t{type.title()} {condition} & {count[condition][type]['images']} & {count[condition][type]['backgrounds']} & {count[condition][type]['instances']} & {count[condition][type]['unique_images']} \\\\")
                print("\t\t\\hline")
            
            # log_str+=f"{type.title()} {condition}: images: {count[condition][type]['images']}; backgrounds: {count[condition][type]['backgrounds']}; labels: {count[condition][type]['instances']}; unique: {len(count[condition][type]['unique_images'])}\n"
        # log_str+=f"{condition}: Unique images between both train/test: {len(set_unique)}/{total_len}\n"
        

    # print(f"\n\n{log_str}")
    # for traintest, data in count.items():
    #     for condition, values in data.items():
    #         print(f"{traintest.title()} {condition}: {values}")

    headers = [""] + list(count['day']['test'].keys())
    # row_header = [f"{key} {sub_key}" for key in count.keys() for sub_key in count[key].keys()]

    # Crear una lista de listas para los datos de la tabla
    table_data = []
    for key, value in count.items():
        for sub_key, sub_value in value.items():
            row = [f"{sub_key} {key}"] + [sub_value[key] for key in sub_value.keys()]
            table_data.append(row)

    # Imprimir la tabla utilizando tabulate
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))

    return count


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


set_info = {'day': {'sets': ['set00', 'set01', 'set02', 'set06', 'set07', 'set08'], 'num_img': 0, 'test': [], 'train': []},
            'night': {'sets': ['set03', 'set04', 'set05', 'set06', 'set10', 'set11'], 'num_img': 0, 'test': [], 'train': []}}


def evaluateInputDataset():
    print()
    for folder_set in os.listdir(kaist_dataset_path):
        set_nun_img = 0
        for subfolder_set in os.listdir(os.path.join(kaist_dataset_path, folder_set)):
            path_set = os.path.join(kaist_dataset_path, folder_set, subfolder_set)
            if os.path.isdir(path_set):
                visible_folder = os.path.join(path_set, visible_folder_name)
                if os.path.exists(visible_folder):
                    nun_img = count_images(visible_folder)
                    set_nun_img += nun_img
                    # print(f"Set: {folder_set}/{subfolder_set} - Num Imags: {nun_img}")
        
        for condition in set_info.values():
            if folder_set in condition['sets']:
                condition['num_img'] += set_nun_img
        print(f"Set: {folder_set} - Num Imags: {set_nun_img}")

    print(f"\nTotal day: {set_info['day']['num_img']}\nTotal night: {set_info['night']['num_img']}\n")

    # print(f"Split img   -> (train - test)")
    # print(f"Day   70-30 -> {int(set_info['day']['num_img']*0.7)} - {int(set_info['day']['num_img']*0.3)}")
    # print(f"Night 70-30 -> {int(set_info['night']['num_img']*0.7)} - {int(set_info['night']['num_img']*0.3)}")
    # print(f"Day   80-20 -> {int(set_info['day']['num_img']*0.8)} - {int(set_info['day']['num_img']*0.2)}")
    # print(f"Night 80-20 -> {int(set_info['night']['num_img']*0.8)} - {int(set_info['night']['num_img']*0.2)}")
    # print(f"Day   90-10 -> {int(set_info['day']['num_img']*0.9)} - {int(set_info['day']['num_img']*0.1)}")
    # print(f"Night 90-10 -> {int(set_info['night']['num_img']*0.9)} - {int(set_info['night']['num_img']*0.1)}")

"""
    Generates set file with percentajes set
"""
def splitDatsets():
    percentajes = [0.7,0.8,0.9]

    sets_img = {}
    for condition, values in set_info.items():
        for folder_set in os.listdir(kaist_dataset_path):
            if folder_set in values['sets']:
                sets_img[folder_set] = []
                for subfolder_set in os.listdir(os.path.join(kaist_dataset_path, folder_set)):
                    path_set = os.path.join(kaist_dataset_path, folder_set, subfolder_set)
                    if os.path.isdir(path_set):
                        visible_folder = os.path.join(path_set, visible_folder_name)
                        if os.path.exists(visible_folder):
                            for root, dirs, files in os.walk(visible_folder):
                                for file in files:
                                    if file.endswith((".jpg", ".jpeg", ".png", ".npy", ".npz")):
                                        img_path_name = os.path.splitext(os.path.join(folder_set, subfolder_set, file))[0]
                                        sets_img[folder_set].append(img_path_name)

        sets_img_ordered = {k: sets_img[k] for k in sorted(sets_img.keys())}
        keys = list(sets_img_ordered.keys())

        # Concatenates training sets, and intercaaltes test sets so that are taken equally
        test_images = [elem for pair in zip_longest(sets_img_ordered[keys[3]] + sets_img_ordered[keys[4]] + sets_img_ordered[keys[5]]) for elem in pair if elem is not None]
        img_list = sets_img_ordered[keys[0]] + sets_img_ordered[keys[1]] + sets_img_ordered[keys[2]] + test_images[::-1]
        
        for percentaje in percentajes:
            # print(f"Datasests for {condition} with training {percentaje} of images are {sets_img_ordered.keys()}")
            values['train'] = []
            values['test'] = []

            n_train_images = len(img_list)*percentaje
            n_test_images = values['num_img'] - n_train_images

            num_added = 0

            for img in img_list:
                if num_added > n_train_images:
                    values['test'] += [img]
                    num_added += 1
                else:
                    values['train'] += [img]
                    num_added += 1
                                                
            file_name = f"kaist_imageSets/train-{condition}-{int(percentaje*100)}_{int(100-percentaje*100)}.txt"
            with open(file_name, 'w') as file:
                for item in values['train']:
                    file.write(item + '\n')
            
            file_name = f"kaist_imageSets/test-{condition}-{int(percentaje*100)}_{int(100-percentaje*100)}.txt"
            with open(file_name, 'w') as file:
                for item in values['test']:
                    file.write(item + '\n')


if __name__ == '__main__':
    # evaluateOutputDataset()
    # evaluateInputDataset()
    # splitDatsets()

    print("\nCase 70-30:")
    evaluateOutputDataset(white_list=['test-day-70_30','train-day-70_30','test-night-70_30','train-night-70_30'], print_latex_table=False)
    print("\nCase 80-20:")
    evaluateOutputDataset(white_list=['test-day-80_20','train-day-80_20','test-night-80_20','train-night-80_20'], print_latex_table=False)
    print("\nCase 90-10:")
    evaluateOutputDataset(white_list=['test-day-90_10','train-day-90_10','test-night-90_10','train-night-90_10'], print_latex_table=False)