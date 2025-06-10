
#!/usr/bin/env python3
# encoding: utf-8

"""
    Script to evaluate dataset image fussion techniques. Time consumption and resulting image :)
"""


import os  
import time
from pathlib import Path
from multiprocessing import Pool, Manager
import concurrent.futures
from collections import defaultdict
from p_tqdm import p_map

import matplotlib.pyplot as plt
import seaborn as sns

import cv2 as cv
import numpy as np

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    from Dataset.fusion_methods.static_image_compression import combine_hsvt, combine_rgbt, combine_vt, combine_vths
    from Dataset.fusion_methods.static_image_compression import combine_rgbt_v2, combine_vths_v2, combine_vths_v3
    from Dataset.fusion_methods.pca_fa_compression import combine_rgbt_fa_to3ch, combine_rgbt_pca_to3ch, combine_rgbt_fa_to1ch, combine_rgbt_pca_to1ch
    from Dataset.fusion_methods.wavelets_mdmr_compression import combine_hsvt_wavelet, combine_rgb_wavelet, combine_hsv_curvelet, combine_rgb_curvelet
    from Dataset.fusion_methods.local_filter_fusion import combine_rgbt_ssim, combine_rgbt_superpixel, combine_rgbt_sobel_weighted
    from utils import color_palette_list, parseYaml, dumpYaml
    from utils.log_utils import logTable
    from utils.color_constants import c_darkgrey,c_grey,c_blue,c_green,c_yellow,c_red,c_purple
    from Dataset_review.review_dataset_kaist import kaist_dataset_path, kaist_yolo_dataset_path, labels_folder_name, images_folder_name, visible_folder_name, lwir_folder_name, store_path
    from Dataset_review.review_dataset_llvip import llvip_yolo_dataset_path


def manager_dict_to_dict(manager_dict):
    """
    Converts manager dict to regular python dict
    """
    standard_dict = {}
    for key, value in manager_dict.items():
        if isinstance(value, Manager().dict().__class__):
            standard_dict[key] = manager_dict_to_dict(value)
        elif isinstance(value, Manager().list().__class__):
            standard_dict[key] = list(value)
        else:
            standard_dict[key] = value
    return standard_dict

def count_images(path):
    images_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".npy", ".npz")):
                images_path.append(file)
    return images_path

"""
    Splittend in function for multiprocessing
"""
def process_dir(args):
    root, directory, visible_folder_name, images_folder_name, unique_images = args
    
    parts = directory.split("-")
    if len(parts) == 3:
        type, condition, id = parts
        visible_path = os.path.join(root, directory, visible_folder_name)
        unique_images[directory]['path'] = visible_path
        unique_images[directory]['img'] = count_images(os.path.join(visible_path, images_folder_name))
    else:
        print(f"Ignoring directory '{directory}' as it doesn't match the expected format.")
        

def process__fussion_image(args):
    rgb_img_path, th_img_path, fussion_function = args
    rgb_img = cv.imread(rgb_img_path)
    th_img = cv.imread(th_img_path, cv.IMREAD_GRAYSCALE)

    start_time = time.perf_counter()
    fussion_function(rgb_img, th_img)
    end_time = time.perf_counter()

    return end_time - start_time

def process_fussion_dir(directory, values, fussion_function):
    visible_path = os.path.join(values['path'], images_folder_name)
    results = [(os.path.join(visible_path, img),
                os.path.join(visible_path.replace(visible_folder_name, lwir_folder_name), img),
                fussion_function)
               for img in values['img']]

    # Procesar imágenes en paralelo usando p_map
    times = p_map(process__fussion_image, results, desc="Processing images", leave=False)

    return times


def plot_distribution(data, mean, std_dev, title="Data Distribution", path = "./figure"):
    plt.figure(figsize=(10, 6))

    # Histogram
    sns.set_palette(color_palette_list)
    sns.histplot(data, kde=True, color=c_blue, stat='density', linewidth=0)

    # Plot settings
    plt.axvline(mean, color=c_red, linestyle='--', label=f'Mean: {mean:.5f}')
    plt.axvline(mean + std_dev, color=c_green, linestyle='--', label=f'Standard Deviation: {std_dev:.5f}')
    plt.axvline(mean - std_dev, color=c_green, linestyle='--')

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')

    plt.legend()

    plt.savefig(path)
    plt.close()

    
"""
    Computes the whole dataset with each fussion function getting the time spent
    on each method for each image
"""
def evaluateFussion(fussion_functions, dataset_whitelist, root = kaist_yolo_dataset_path, store_path = store_path):
    file_log_path = os.path.join(store_path, 'fussion_data.yaml')
    
    if not os.path.exists(file_log_path):
        manager = Manager()
        unique_images = manager.dict()
        for dataset in dataset_whitelist:
            unique_images[dataset] = manager.dict()
            unique_images[dataset]['path'] = str()
            unique_images[dataset]['img'] = []

        print(f"Get dir path to be processed from {root}")
        # Get all dirs to process
        folder_path_list = []
        for directory_name in os.listdir(root):
            directory = os.path.join(root, directory_name)
            if os.path.isdir(directory):
                if directory_name.startswith(("test", "train")):
                    if not dataset_whitelist or directory_name in dataset_whitelist:
                        folder_path_list.append((root, directory_name, visible_folder_name, images_folder_name, unique_images))
                    else:
                        pass
        
        print(f"{dataset_whitelist = }")
        with Pool() as pool:
            pool.map(process_dir, folder_path_list)

        images_dict = manager_dict_to_dict(unique_images)
        dumpYaml(file_path=file_log_path, data=images_dict)

    images_dict = parseYaml(file_path=file_log_path)

    for index, (directory, values) in enumerate(images_dict.items()):
        print(f"[{index+1}/{len(images_dict.items())}] Processing directory: {directory}")

        # Aplicar cada función de fusión en paralelo para cada directorio
        for index, fussion_function in enumerate(fussion_functions):
            
            if fussion_function.__name__ in values:
                print(f"\t[{index+1}/{len(fussion_functions)}] {fussion_function.__name__} loaded from YAML")
                continue
            
            print(f"\t[{index+1}/{len(fussion_functions)}] Applying fusion function: {fussion_function.__name__}")

            # Procesar el directorio con la función de fusión actual
            times = process_fussion_dir(directory, values, fussion_function)
            values[f'{fussion_function.__name__}'] = times

            # Volcar los resultados a un archivo YAML
            dumpYaml(file_path=file_log_path, data=images_dict)

    # Compute Average, deviation and distribution
    images_store_path = os.path.join(store_path, 'png')
    if not os.path.exists(images_store_path):
        os.makedirs(images_store_path)

    table_headers = ["Fussion Function", "Mean (s)", "Std (s)"]
    table_data = []
    for index, fussion_function in enumerate(fussion_functions):
        data = []
        for index, (directory, values) in enumerate(images_dict.items()):
            if fussion_function.__name__ in values:
                data.extend(values[f'{fussion_function.__name__}'])

        data = values[f'{fussion_function.__name__}']
        mean = np.mean(data)
        std_dev = np.std(data)
        path = os.path.join(images_store_path, f'{fussion_function.__name__}.png')
        plot_distribution(data, mean, std_dev, title=f"{fussion_function.__name__} for {directory}", path = path)
        
        row = [f"{fussion_function.__name__.title()}", f"{mean:.5f}", f"{std_dev:.5f}"]
        table_data.append(row)
    
    logTable(table_data, store_path, 'fussion_table')

"""
    Creates a demo image of the result of applying the fussion functions provided
    to a given image
"""
def splitSingleImage(fussion_functions, store_path=store_path, 
                     img_path='/home/arvc/eeha/kaist-cvpr15/images/set00/V000/lwir/I01689.jpg'):
    if not os.path.exists(os.path.join(store_path, 'split')):
        os.makedirs(os.path.join(store_path, 'split'))
    
    img_lwir = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_visible = cv.imread(img_path.replace('/lwir/','/visible/'))


    b_channel, g_channel, r_channel = cv.split(img_visible)
    image_hsv = cv.cvtColor(img_visible, cv.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv.split(image_hsv)

    zeros = np.zeros_like(b_channel)

    cv.imwrite(os.path.join(store_path,'split','original_visible_image.png'), img_visible)
    cv.imwrite(os.path.join(store_path,'split','original_lwir_image.png'), img_lwir)
    cv.imwrite(os.path.join(store_path,'split','B_channel.png'), cv.merge([b_channel, zeros, zeros]))
    cv.imwrite(os.path.join(store_path,'split','G_channel.png'), cv.merge([zeros, g_channel, zeros]))
    cv.imwrite(os.path.join(store_path,'split','R_channel.png'), cv.merge([zeros, zeros, r_channel]))
    cv.imwrite(os.path.join(store_path,'split','H_channel.png'), cv.applyColorMap(h_channel, cv.COLORMAP_HSV))
    cv.imwrite(os.path.join(store_path,'split','S_channel.png'), s_channel)
    cv.imwrite(os.path.join(store_path,'split','V_channel.png'), v_channel)
    cv.imwrite(os.path.join(store_path,'split','LWIR_channel.png'), cv.applyColorMap(img_lwir, cv.COLORMAP_JET))

    for fussion in fussion_functions:
        path = os.path.join(store_path,'split',f'{fussion.__name__}.png')
        image = fussion(img_visible, img_lwir)
        cv.imwrite(path, image)

    print(f"Stored fussion data in {os.path.join(store_path)}")

if __name__ == '__main__':

    kaist_store_path = os.path.join(store_path, 'fussion')
    if not os.path.exists(kaist_store_path):
        os.makedirs(kaist_store_path)

    evaluate_functions = [combine_hsvt, combine_rgbt, combine_vt, combine_vths,
                          combine_rgbt_v2, combine_vths_v2, combine_vths_v3,
                          combine_rgbt_fa_to3ch, combine_rgbt_pca_to3ch, 
                        #   combine_rgbt_fa_to1ch, combine_rgbt_pca_to1ch,
                          combine_rgb_wavelet, combine_rgb_curvelet, #combine_hsvt_wavelet, combine_hsv_curvelet, 
                          combine_rgbt_ssim, combine_rgbt_superpixel, combine_rgbt_sobel_weighted
                         ]

    # splitSingleImage(fussion_functions=evaluate_functions, store_path=kaist_store_path)

    # print("\nCase 80-20:")
    # white_list=['test-day-80_20','train-day-80_20','test-night-80_20','train-night-80_20']
    # evaluateFussion(fussion_functions=evaluate_functions,dataset_whitelist=white_list, store_path=kaist_store_path)


    ## For LLVIP dataset
    from Dataset_review.review_dataset_llvip import store_path
    
    llvip_store_path = os.path.join(store_path, 'fussion')
    if not os.path.exists(llvip_store_path):
        os.makedirs(llvip_store_path)

    splitSingleImage(fussion_functions=evaluate_functions, store_path=llvip_store_path,
                     img_path='/home/arvc/eeha/LLVIP/visible/test/210263.jpg')

    # print("\nCase 80-20:")
    # white_list=['test-night-80_20','train-night-80_20']
    # evaluateFussion(fussion_functions=evaluate_functions,dataset_whitelist=white_list, root=llvip_yolo_dataset_path, store_path=llvip_store_path)
