
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
from matplotlib import cm
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
set_info_ = {'day': {'sets': ['set00', 'set01', 'set02', 'set06', 'set07', 'set08'], 'num_images': 0, 'heatmap': [None,None,None,None], 'CLAHE heatmap': [None,None,None,None]},
            'night': {'sets': ['set03', 'set04', 'set05', 'set09', 'set10', 'set11'], 'num_images': 0, 'heatmap': [None,None,None,None], 'CLAHE heatmap': [None,None,None,None]},
            'day+night': {'num_images': 0, 'heatmap': [None,None,None,None], 'CLAHE heatmap': [None,None,None,None]}}
            
set_info =  {'BGR': copy.deepcopy(set_info_), 'HSV': copy.deepcopy(set_info_)}    

store_path_heatheatmap = os.path.join(store_path, 'review_heatheatmap')

def rescale_channel_minmax(channel, min_value=None, max_value=None, new_min=0, new_max=255, mask=None):
    channel_rescaled = channel.copy().astype(np.float32)
    
    if mask is not None:
        mask_index = mask > 0
    else:
        mask_index = np.ones(channel_rescaled.shape, dtype=bool)

    if min_value is None:
        min_value = np.min(channel_rescaled[mask_index])    

    if max_value is None:
        max_value = np.max(channel_rescaled[mask_index])   

    # Set average value to masked parts so that they do not interfere later
    average = np.average(channel_rescaled[mask_index])
    channel_rescaled[~mask_index] = average

    if max_value == min_value:
        print(f"[ERROR] Min and max value are the same ({min_value = }; {max_value = })")
        channel_rescaled[mask_index] = (new_min + new_max) / 2
        return channel_rescaled.astype(np.uint8), min_value, max_value

    channel_rescaled = (channel_rescaled - min_value) / (max_value - min_value)  # Normalize a [0, 1]
    channel_rescaled = channel_rescaled * (new_max - new_min) + new_min          # Escalar a [new_min, new_max]
        
    # Asegurarse de que los valores estén dentro del rango [new_min, new_max]
    channel_rescaled = np.clip(channel_rescaled, new_min, new_max)
        
    # Convertir de nuevo a uint8 si los nuevos valores están en el rango 0-255
    if new_min >= 0 and new_max <= 255:
        channel_rescaled = channel_rescaled.astype(np.uint8)

    return channel_rescaled, min_value, max_value

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
    if all_heatmaps is None:
        all_heatmaps = [None, None, None, None]
        
    for i in range(4):  
        if all_heatmaps[i] is None:
            all_heatmaps[i] = np.zeros_like(image_heatmaps[0]).astype(float)

        all_heatmaps[i] += image_heatmaps[i].astype(float) / float(n_images)   
    
    return all_heatmaps

def process_images(args):
    path, heatheatmap_channel_config = args
    heatmap = [None,None,None,None]
    eq_heatmap = [None,None,None,None]

    n_images = sum(1 for file in os.listdir(path) if file.endswith((".jpg", ".jpeg", ".pdf", ".npy", ".npz")))
    for file in os.listdir(path):
        if file.endswith((".jpg", ".jpeg", ".pdf", ".npy", ".npz")):
            img_path = os.path.join(path, file)
            ret_heatmap, ret_eq_heatmap = extract_heatmap(img_path, heatheatmap_channel_config)

            heatmap = updateHeatmapList(heatmap, ret_heatmap, n_images)
            eq_heatmap = updateHeatmapList(eq_heatmap, ret_eq_heatmap, n_images)
    
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
                #         break
                #     break
                # break
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing folders'):
                results.append(future.result())

            for result in results:
                heatmap, eq_heatmap, path, heatheatmap_channel_config, n_images = result
                tag = heatheatmap_channel_config['tag']
                for condition in ['day', 'night']:
                    if isPathCondition(set_info[tag][condition]['sets'], path):
                        set_info[tag][condition]['num_images']+=n_images
                        break
                
            for result in results:
                heatmap, eq_heatmap, path, heatheatmap_channel_config, n_images = result
                tag = heatheatmap_channel_config['tag']
                for condition in ['day', 'night']:
                    if isPathCondition(set_info[tag][condition]['sets'], path):
                        # cv.imwrite(os.path.join(store_path_heatheatmap, f"heatmap[0]_{n_images}.png"), heatmap[0].astype(np.uint8)) 
                        # cv.imwrite(os.path.join(store_path_heatheatmap, f"heatmap[1]_{n_images}.png"), heatmap[1].astype(np.uint8)) 

                        rescaled_heatmap = [heatmap_ch*n_images for heatmap_ch in heatmap]
                        rescaled_eq_heatmap = [heatmap_ch*n_images for heatmap_ch in eq_heatmap]
                        set_info[tag][condition]['heatmap'] = updateHeatmapList(set_info[tag][condition]['heatmap'], rescaled_heatmap, set_info[tag][condition]['num_images'])
                        set_info[tag][condition]['CLAHE heatmap'] = updateHeatmapList(set_info[tag][condition]['CLAHE heatmap'], rescaled_eq_heatmap, set_info[tag][condition]['num_images'])

                        # img_data,_ ,_ = rescale_channel_minmax(set_info[tag][condition]['heatmap'][0])
                        # img_data2,_ ,_ = rescale_channel_minmax(set_info[tag]['day']['heatmap'][1])
                        # cv.imwrite(os.path.join(store_path_heatheatmap, f"{tag}_{condition}_heatmap[0]_{set_info[tag][condition]['num_images']}.png"), img_data) 
                        # cv.imwrite(os.path.join(store_path_heatheatmap, f"{tag}_{condition}_heatmap[1]_{set_info[tag][condition]['num_images']}.png"), img_data2) 
                        break
                            
        with open(cache_file_path, 'wb') as f:
            pickle.dump(set_info, f)

    else:    
        print(f'Reload previous heatmap data stored')
        with open(cache_file_path, 'rb') as f:
            set_info = pickle.load(f)

    for heatmap_type in ['heatmap']: #, 'CLAHE heatmap']: 
        for heatheatmap_channel_cfg in heatheatmap_channel_cfg_list:
            tag = heatheatmap_channel_cfg['tag']
            for condition in ['day', 'night']:

                channel_names = heatheatmap_channel_cfg['channel_names'] + ['LWIR']
                for ch in range(len(channel_names)):
                    
                    img_data = set_info[tag][condition][heatmap_type][ch]
                    if img_data is None:
                        continue
                    file_name = f'{heatmap_type}_{condition}_{channel_names[ch]}'
                    file_name_png = os.path.join(store_path_heatheatmap, tag, f'{file_name}.png')
                    img_data,_ ,_ = rescale_channel_minmax(img_data)
                    colored_img = cv.applyColorMap(img_data, cv.COLORMAP_JET)
                    
                    cv.imwrite(file_name_png.replace('.png','_bn.png'), img_data)
                    cv.imwrite(file_name_png, colored_img)

                    fig, ax = plt.subplots(figsize=(6, 6))

                    cax = ax.imshow(img_data, cmap='jet', vmin=np.min(img_data), vmax=np.max(img_data))
                    ax.axis('off')
                    cbar = plt.colorbar(cax, ax=ax, orientation='vertical', fraction=0.037, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
                    # cbar.set_label('Intensity')

                    plt.subplots_adjust(right=0.85)
                    plt.title(f"{channel_names[ch]} Channel ({condition} with {set_info[tag][condition]['num_images']} images)", fontsize=10)
                    plt.savefig(file_name_png.replace('png', 'pdf'), bbox_inches='tight', pad_inches=0.05)
                    plt.close()
            

if __name__ == '__main__':

    for tag in [item['tag'] for item in heatheatmap_channel_cfg_list]:
        os.makedirs(os.path.join(store_path_heatheatmap, tag), exist_ok=True)
            
    evaluateInputDataset()
    print(f"Finished heatmap generation, stored data in {store_path_heatheatmap}")
