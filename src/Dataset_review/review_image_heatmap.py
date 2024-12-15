
#!/usr/bin/env python3
# encoding: utf-8

"""
    Script to evaluate image heatmaporgams and the effects of the equalization on them
    on single images and in all dataset
"""

import os  
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from tqdm import tqdm
from tabulate import tabulate

import math
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm 
from matplotlib.ticker import LogLocator
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

import pickle
import cv2 as cv      

# Small hack so packages can be found
# if __name__ == "__main__":
import sys
sys.path.append('./src')
from utils.color_constants import c_darkgrey,c_grey,c_blue,c_green,c_yellow,c_red,c_purple
from Dataset_review.review_dataset import home, kaist_dataset_path, visible_folder_name, lwir_folder_name, store_path

# Careful when chosing y_limit if working with or without normalized heatheatmaps or not!
heatheatmap_channel_cfg_list = [{'tag': 'BGR', 'conversion': None, 
                               'channel_names': ['B', 'G', 'R'], 'y_limit': [15,10,12.5,30]}, 
                              {'tag': 'HSV', 'conversion': cv.COLOR_BGR2HSV_FULL, 
                               'channel_names': ['H', 'S', 'V'], 'y_limit': [17,11,11,30]}]

# Store heatheatmaps in a list of [b,g,r,lwir] heatmaps for each image
set_info_ = {'day': {'sets': ['set00', 'set01', 'set02', 'set06', 'set07', 'set08'], 'heatmap': None},
            'night': {'sets': ['set03', 'set04', 'set05', 'set09', 'set10', 'set11'], 'heatmap': None},
            'day+night': {'heatmap': None}}
            
set_info =  {'BGR': copy.deepcopy(set_info_), 'HSV': copy.deepcopy(set_info_)}    

store_path_heatheatmap = os.path.join(store_path, 'review_heatheatmap')


def readImage(path, flag = None):
    if flag is not None:
        imagen = cv.imread(path, flag)
    else:
        imagen = cv.imread(path)
    img_heigth, img_width = imagen.shape[:2]
    aspect_ratio = img_width / img_heigth
    cut_pixels_width = 0
    cut_pixels_height = int(cut_pixels_width/aspect_ratio)
    imagen_recortada = imagen[cut_pixels_height:img_heigth-cut_pixels_height, cut_pixels_width:img_width-cut_pixels_width]
    imagen_final = cv.resize(imagen_recortada, (img_width, img_heigth))
    
    return imagen_final

def save_images(img_heatmap_list, filename, img_path, store_path=store_path_heatheatmap):
    rows = len(img_heatmap_list)
    cols = 2

    item = 1
    plt.figure(figsize=[cols*6.4, rows*5.12])
    plt.rc('font', size=14)
    for col_data in img_heatmap_list:
        plt.subplot(rows, cols, item)
        plt.imshow(col_data['img'])
        plt.title(col_data['img_title'], fontsize=20)
        plt.axis('off')

        plt.subplot(rows, cols, item+1)
        plt.plot(col_data['heatmap'], color = '#F6AE2D')
        plt.title(col_data['heatmap_title'], fontsize=20)
        # plt.axis('off')
        item+=2   
    
    plt.annotate(f'Img: {img_path}',
                    xy = (1.0, -0.1), xycoords='axes fraction',
                    ha='right', va="center", fontsize=14,
                    color='black', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(store_path,'heatheatmap_pdf',f'{filename}.pdf'))
    plt.savefig(os.path.join(store_path,'heatheatmap_png',f'{filename}.png'))
    plt.close()


def getheatmapEqCLAHE(img_channel):
    clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(6,6))
    eq_img = clahe.apply(img_channel)
    
    return eq_img


def extract_heatmap(img_path, heatheatmap_channel_config, plot = False):
    ch0, ch1, ch2, lwir = [], [], [], []
    eq_ch0, eq_ch1, eq_ch2, eq_lwir = [], [], [], []

    # Process visible image
    img = readImage(img_path)

    assert img is not None, "file could not be read, check with os.path.exists()"

    if heatheatmap_channel_config['conversion'] is not None:
        image = cv.cvtColor(img, heatheatmap_channel_config['conversion'])
    else:
        image = img
        
    ch0,ch1,ch2 = cv.split(image)
    
    ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = getheatmapEqCLAHE(ycrcb_img[:, :, 0]) 
    equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)

    if heatheatmap_channel_config['conversion'] is not None:
        equalized_image = cv.cvtColor(equalized_img, heatheatmap_channel_config['conversion'])
    else:
        equalized_image = equalized_img
        
    eq_ch0,eq_ch1,eq_ch2 = cv.split(equalized_img)

    ## not elegant at all... :) sorryn't, for now...
    # only LWIR when processing Kaist dataset
    if __name__ == '__main__':
        # Process LWIR correspondant image
        img_path = img_path.replace(visible_folder_name, lwir_folder_name)
        lwir = readImage(img_path, cv.IMREAD_GRAYSCALE)
        assert lwir is not None, f"File could not be read, check with os.path.exists() that {img_path} exists"
        
        eq_lwir = getheatmapEqCLAHE(lwir)
        
    return [ch0, ch1, ch2, lwir], [eq_ch0, eq_ch1, eq_ch2, eq_lwir]


def updateHeatmapList(all_heatmaps, image_heatmaps, n_images):
    for i in range(4):  
        if all_heatmaps[i] is None:
            all_heatmaps[i] = np.zeros_like(image_heatmaps[0]).astype(float)

        single_heatmap = image_heatmaps[i].astype(float) / n_images        
        all_heatmaps[i] += single_heatmap

def process_images(args):
    path, heatheatmap_channel_config = args
    heatmap = [None,None,None,None]
    eq_heatmap = [None,None,None,None]

    n_images = sum(1 for file in os.listdir(path) if file.endswith((".jpg", ".jpeg", ".pdf", ".npy", ".npz")))
    for file in os.listdir(path):
        if file.endswith((".jpg", ".jpeg", ".pdf", ".npy", ".npz")):
            img_path = os.path.join(path, file)
            ret_heatmap, ret_eq_heatmap = extract_heatmap(img_path, heatheatmap_channel_config)

            updateHeatmapList(heatmap, ret_heatmap, n_images)
            updateHeatmapList(eq_heatmap, ret_eq_heatmap, n_images)
    
    return heatmap, eq_heatmap, path, heatheatmap_channel_config, n_images

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


def evaluateInputDataset():
    global set_info

    cache_file_path = os.path.join(store_path_heatheatmap, 'set_info.pkl')
    if not os.path.exists(cache_file_path):
        with ThreadPoolExecutor() as executor:
            futures = []
            for heatheatmap_channel_config in heatheatmap_channel_cfg_list:
                for folder_set in os.listdir(kaist_dataset_path):
                    for subfolder_set in os.listdir(os.path.join(kaist_dataset_path, folder_set)):
                        path_set = os.path.join(kaist_dataset_path, folder_set, subfolder_set)
                        if os.path.isdir(path_set):
                            visible_folder = os.path.join(path_set, visible_folder_name)
                            if os.path.exists(visible_folder):
                                args = visible_folder, heatheatmap_channel_config
                                futures.append(executor.submit(process_images, args))
                #     break
                # break

            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing folders'):
                heatmap, eq_heatmap, path, heatheatmap_channel_config, n_images = future.result()
                tag = heatheatmap_channel_config['tag']
                for condition in ['day', 'night']:
                    if isPathCondition(set_info[tag][condition]['sets'], path):
                        updateHeatmapList(set_info[tag][condition]['heatmap'], heatmap, n_images)
                        updateHeatmapList(set_info[tag][condition]['CLAHE heatmap'], eq_heatmap, n_images)
                        break
        
        with open(cache_file_path, 'wb') as f:
            pickle.dump(set_info, f)

    else:    
        print(f'Reload previous heatmap data stored')
        with open(cache_file_path, 'rb') as f:
            set_info = pickle.load(f)

    for heatmap_type in ['heatmap', 'CLAHE heatmap']: 
        for heatheatmap_channel_cfg in heatheatmap_channel_cfg_list:
            tag = heatheatmap_channel_cfg['tag']
            for condition in ['day', 'night']:

                channel_names = heatheatmap_channel_cfg['channel_names'] + ['LWIR']
                for ch in range(len(channel_names)):
                    
                    img_data = set_info[tag][condition][heatmap_type][ch]

                    file_name = f'{heatmap_type}_{condition}_{channel_names[ch]}'
                    file_name_pdf = os.path.join(store_path_heatheatmap,'heatheatmap_pdf', f'{file_name}.pdf')
                    file_name_png = os.path.join(store_path_heatheatmap,'heatheatmap_png', f'{file_name}.png')
                    
                    cv.imwrite(file_name_pdf, img_data)
                    cv.imwrite(file_name_png, img_data)
            

if __name__ == '__main__':

    for heatmap_format in ['heatheatmap_pdf', 'heatheatmap_png']:
        for tag in [item['tag'] for item in heatheatmap_channel_cfg_list]:
            os.makedirs(os.path.join(store_path_heatheatmap, heatmap_format, tag), exist_ok=True)
            os.makedirs(os.path.join(store_path_heatheatmap, heatmap_format, f'{tag}_log_scale'), exist_ok=True)


    evaluateInputDataset()
