
#!/usr/bin/env python3
# encoding: utf-8

"""
    Script to evaluate image historgams and the effects of the equalization on them
    on single images and in all dataset
"""

import os  
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tabulate import tabulate

import numpy as np

import pickle
import cv2 as cv      

# Small hack so packages can be found
# if __name__ == "__main__":
import sys
sys.path.append('./src')
from Dataset_review.review_dataset import kaist_dataset_path, visible_folder_name, lwir_folder_name, store_path
from Dataset_review.review_histograms import readImage

# Store histograms in a list of [b,g,r,lwir] hists for each image
set_info = {'day': {'sets': ['set00', 'set01', 'set02', 'set06', 'set07', 'set08'], 'contrast': [], 'sharpness': []},
            'night': {'sets': ['set03', 'set04', 'set05', 'set09', 'set10', 'set11'], 'contrast': [], 'sharpness': []},
            'day+night': {'contrast': [], 'sharpness': []}}
            

store_path_imagem = os.path.join(store_path, 'image_metrics')


def extract_imag_metrics(img_path, plot = False):
    contrast, sharpness, contrast_lwir, sharpness_lwir = 0,0,0,0

    # Process visible image
    img = readImage(img_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    def computeContrast(image):
        N, M = image.shape
        B = np.mean(image)
        return np.sqrt(np.sum((image - B) ** 2) / (N * M))

    def computeSharpness(image):
        N, M = image.shape
        sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

        # Max change direction
        Dn = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return (np.sum(Dn ** 2) / (N * M))*100


    contrast = computeContrast(img)
    sharpness = computeSharpness(img)

    ## not elegant at all... :) sorryn't, for now...
    # only LWIR when processing Kaist dataset
    if __name__ == '__main__':
        # Process LWIR correspondant image
        img_path = img_path.replace(visible_folder_name, lwir_folder_name)
        img = readImage(img_path, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        
        contrast_lwir = computeContrast(img)
        sharpness_lwir = computeSharpness(img)
        
    return [contrast, contrast_lwir], [sharpness, sharpness_lwir]


def process_images(path):
    contrast = []
    sharpness = []
    for file in os.listdir(path):
        if file.endswith((".jpg", ".jpeg", ".pdf", ".npy", ".npz")):
            img_path = os.path.join(path, file)
            ret_contrast, ret_sharpness = extract_imag_metrics(img_path)

            contrast.append(ret_contrast)
            sharpness.append(ret_sharpness)

    return contrast, sharpness, path

# Condition is day/night
def isPathCondition(set_info, path):
    for img_set in set_info:
        if img_set in path:
            return True
    return False

# Compute shape of list of lists :)
def shape(lista):
    if isinstance(lista, list):
        return [len(lista)] + shape(lista[0]) if lista else []
    elif isinstance(lista, np.ndarray):
        return list(lista.shape)
    else:
        return []


def computeChannelMetrics(datach,log_table):
    data_ch = np.array(datach)
    
    data_mean = np.mean(data_ch)
    data_std = np.std(data_ch)
    variation_coef =  (data_std / data_mean) * 100
    
    if log_table is not None:
        log_table[-1].extend([f"{variation_coef:.3f}",
                              f"{data_mean:.2f}",
                              f"{data_std:.3f}",
                              f"{len(data_ch)}"])
                

def reviewImageMetrics():
    global set_info

    cache_file_path = os.path.join(store_path_imagem, 'image_info_metrics.pkl')
    if not os.path.exists(cache_file_path):
        with ThreadPoolExecutor() as executor:
            futures = []
            for folder_set in os.listdir(kaist_dataset_path):
                for subfolder_set in os.listdir(os.path.join(kaist_dataset_path, folder_set)):
                    path_set = os.path.join(kaist_dataset_path, folder_set, subfolder_set)
                    if os.path.isdir(path_set):
                        visible_folder = os.path.join(path_set, visible_folder_name)
                        if os.path.exists(visible_folder):
                            futures.append(executor.submit(process_images, visible_folder))
            #     break
            # break

            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing folders'):
                contrast, sharpness, path = future.result()
                for condition in ['day', 'night']:
                    if isPathCondition(set_info[condition]['sets'], path):
                        set_info[condition]['contrast'].extend(contrast)
                        set_info[condition]['sharpness'].extend(sharpness)
                        break
        
        with open(cache_file_path, 'wb') as f:
            pickle.dump(set_info, f)

    else:    
        print(f'Reload previous hist data stored')
        with open(cache_file_path, 'rb') as f:
            set_info = pickle.load(f)


    for condition in ['day', 'night']:
        for data_type in ['contrast', 'sharpness']:
            ## Accumulate day+night data
            # set_info['day+night'][data_type].extend(set_info[condition][data_type])

            # Theres n_images array of 2 elements, we want 2 arrays of data. Need to transpose :)
            four_channel_hist_list = [[],[]]
            for i in range(2):
                for channel in set_info[condition][data_type]:
                    four_channel_hist_list[i].append(channel[i])

            set_info[condition][data_type] = four_channel_hist_list
 
    for condition in ['day', 'night']: #, 'day+night']:
        for data_type in ['contrast', 'sharpness']:
            data = set_info[condition][data_type]
            channel_names = ['Visible','LWIR']
            log_table_headers = ['Test', 'CV', 'Mean', 'Std.', 'N Img.']
            log_table_data = []
            for ch in range(2):
                log_table_data.append([f"[{condition}][{data_type}][{channel_names[ch]}]"])
                computeChannelMetrics(data[ch], log_table_data)            

            print(tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty"))
        

if __name__ == '__main__':
    os.makedirs(os.path.join(store_path_imagem), exist_ok=True)
    reviewImageMetrics()
