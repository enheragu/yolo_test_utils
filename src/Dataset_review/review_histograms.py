
#!/usr/bin/env python3
# encoding: utf-8

import os  
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from itertools import zip_longest
import copy
from tabulate import tabulate
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

save_lock = threading.Lock()

home = Path.home()
kaist_dataset_path = f"{home}/eeha/kaist-cvpr15/images"
kaist_yolo_dataset_path = f"{home}/eeha/kaist-yolo-annotated/"
labels_folder_name = "labels"
images_folder_name = "images"
visible_folder_name = "visible"
lwir_folder_name = "lwir"

store_path = f"{home}/eeha/dataset_analysis/"


def save_images(img1, img2, hist1, hist2, filename, img_path):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title('Eq-Histogram Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.plot(hist1, color = '#F6AE2D')
    plt.title('Original histogram')

    plt.subplot(2, 2, 4)
    plt.plot(hist2, color = '#F6AE2D')
    plt.title('Eq Histogram')
    
    plt.tight_layout()

    plt.annotate(f'Img: {img_path}',
                    xy = (1.0, -0.17), xycoords='axes fraction',
                    ha='right', va="center", fontsize=8,
                    color='black', alpha=0.2)
    
    plt.savefig(os.path.join(store_path,'pdf',f'{filename}.pdf'))
    plt.savefig(os.path.join(store_path,'png',f'{filename}.png'))

def gethistEqCLAHE(img):
    clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(6,6))
    eq_img = clahe.apply(img)
    clahe_hist = cv.calcHist([eq_img], [0], None, [256], [0, 256])

    return eq_img, clahe_hist

def gethistEq(img):
    eq_img = cv.equalizeHist(img) # Enhances noise too!
    hist = cv.calcHist([eq_img], [0], None, [256], [0, 256])

    return eq_img, hist


def gethistExpLinear(img):

def extract_hist(img_path, plot = False):
    b, g, r, lwir = [], [], [], []
    eq_b, eq_g, eq_r, eq_lwir = [], [], [], []

    # Process visible image
    img = cv.imread(img_path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    b = cv.calcHist([img], [0], None, [256], [0, 256])
    g = cv.calcHist([img], [1], None, [256], [0, 256])
    r = cv.calcHist([img], [2], None, [256], [0, 256])
    
    b_ch,g_ch,r_ch = cv.split(img)
    eq_b = gethistEqCLAHE(b_ch)[1]
    eq_g = gethistEqCLAHE(g_ch)[1]
    eq_r = gethistEqCLAHE(r_ch)[1]

    # Process LWIR correspondant image
    img_path = img_path.replace(visible_folder_name, lwir_folder_name)
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    lwir = cv.calcHist([img], [0], None, [256], [0, 256])
    eq_lwir = gethistEqCLAHE(img)[1]
        
    return [b, g, r, lwir], [eq_b, eq_g, eq_r, eq_lwir]


def process_image(path):
    hist = []
    eq_hist = []
    for file in os.listdir(path):
        if file.endswith((".jpg", ".jpeg", ".pdf", ".npy", ".npz")):
            img_path = os.path.join(path, file)
            ret_hist, ret_eq_hist = extract_hist(img_path)

            hist.append(ret_hist)
            eq_hist.append(ret_eq_hist)

    return hist, eq_hist, path



def save_histogram_image(hist, title, filename, color, n_images, log_scale = False, condition = '-'):
    plt.figure(figsize=(4.5, 3))
    # plt.plot(hist, color = color)
    # plt.fill_between(np.arange(len(hist)), hist, color=color, alpha=0.3)
    # plt.hist(hist, bins=range(len(hist)), color=color, alpha=0.7)

    plt.plot(hist[1], color = color, label = f"{title} (max)")
    # plt.plot(hist[0], color = color)
    # plt.fill_between(np.arange(len(hist[0])), hist[0], color=color, alpha=0.8, label = "min") # between 0 and min
    plt.fill_between(np.arange(len(hist[0])), hist[1], color=color, alpha=0.3) #, label = "variance") # between 0 and max
    if log_scale: plt.yscale('log')

    plt.title(f'{title} histogram ({n_images} images) ({condition} condition)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency' + (' (log scale)' if log_scale else ''))
    
    plt.tight_layout()
    plt.savefig(os.path.join(store_path,'pdf',f'{filename}.pdf'))
    plt.savefig(os.path.join(store_path,'png',f'{filename}.png'))
    plt.close()


def plot_histograms(b_hist, g_hist, r_hist, lwir_hist, titles, colors, filename = 'histograms', n_images = 0, log_scale = False, condition = '-'):
    plt.figure(figsize=(9, 3))

    # First subplot
    plt.subplot(1, 2, 1)
    for hist, title, color in zip([b_hist, g_hist, r_hist], titles, colors):
        # plt.hist(channel, bins=range(len(channel)), color=color, alpha=0.7)
        plt.plot(hist[1], color = color, label = f"{title} (max)")
        # plt.plot(hist[0], color = color)
        # plt.fill_between(np.arange(len(hist[0])), hist[0], color=color, alpha=0.8, label = "min histogram") # between 0 and min
        plt.fill_between(np.arange(len(hist[0])), hist[1], color=color, alpha=0.3) #, label = "variance") # between 0 and max
        if log_scale: plt.yscale('log')
        # Single histogram for each BGR channel
        # save_histogram_image(hist, title, f'{title.lower().replace(" ", "_")}_{filename}', color=color, n_images=n_images, log_scale=log_scale)
    
    plt.legend()
    plt.title(f'RGB histograms ({n_images} images) ({condition})')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency' + (' (log scale)' if log_scale else ''))

    # Second subplot
    plt.subplot(1, 2, 2)
    plt.plot(lwir_hist[1], color = colors[-1], label = f"{titles[-1]} (max)")
    # plt.plot(lwir_hist[0], color = colors[-1])
    # plt.fill_between(np.arange(len(lwir_hist[0])), lwir_hist[0], color=colors[-1], alpha=0.8, label = "min histogram") # between 0 and min
    plt.fill_between(np.arange(len(lwir_hist[0])), lwir_hist[1], color=colors[-1], alpha=0.3) #, label = "variance") # Minimum in between 0 and max
    if log_scale: plt.yscale('log')
    save_histogram_image(lwir_hist, titles[-1], f'{titles[-1].lower().replace(" ", "_")}_{filename}', color=colors[-1], n_images = n_images, log_scale=log_scale)
    plt.legend()
    plt.title(f'LWIR histogram ({n_images} images) ({condition})')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency' + (' (log scale)' if log_scale else ''))

    plt.tight_layout()
    plt.savefig(os.path.join(store_path,'pdf',f'{filename}.pdf'))
    plt.savefig(os.path.join(store_path,'png',f'{filename}.png'))
    # plt.show()

# Store histograms in a list of [b,g,r,lwir] hists for each image
set_info = {'day': {'sets': ['set00', 'set01', 'set02', 'set06', 'set07', 'set08'], 'hist': [], 'CLAHE hist': []},
            'night': {'sets': ['set03', 'set04', 'set05', 'set06', 'set10', 'set11'], 'hist': [], 'CLAHE hist': []},
            'day+night': {'hist': [], 'CLAHE hist': []}}


# Condition is day/night
def isPathCondition(condition, path):
    for img_set in set_info[condition]['sets']:
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

    with ThreadPoolExecutor() as executor:
        futures = []
        for folder_set in os.listdir(kaist_dataset_path):
            for subfolder_set in os.listdir(os.path.join(kaist_dataset_path, folder_set)):
                path_set = os.path.join(kaist_dataset_path, folder_set, subfolder_set)
                if os.path.isdir(path_set):
                    visible_folder = os.path.join(path_set, visible_folder_name)
                    if os.path.exists(visible_folder):
                        futures.append(executor.submit(process_image, visible_folder))
                # break
            # break

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing imageSet folders'):
            hist, eq_hist, path = future.result()
            for condition in ['day', 'night']:
                if isPathCondition(condition, path):
                    set_info[condition]['hist'].extend(hist)
                    set_info[condition]['CLAHE hist'].extend(eq_hist)
                    break
    

    
    def plotAccumulatedHist(condition, data, type):
        hist_data = []
        for ch in range(4):
            min_vals = np.min(data[ch], axis=0)[:,0]
            max_vals = np.max(data[ch], axis=0)[:,0]
            hist_data.append([min_vals, max_vals])
        
        if isinstance(type, list):
            b_type, g_type, r_type, lwir_type = type
            type = 'combined'
        else:
            b_type, g_type, r_type, lwir_type = type, type, type, type


        for log_scale in [True, False]:
            plot_histograms(hist_data[0], hist_data[1], hist_data[2], hist_data[3],
                            [f"B {b_type}",f"G {g_type}",f"R {r_type}",f"LWIR {lwir_type}"], 
                            ['#0171BA', '#78B01C', '#F23535', '#F6AE2D'],
                            f'{"log_" if log_scale else ""}{condition}_{type}',
                            n_images=len(data[ch]), log_scale=log_scale, condition = condition)

             
    for condition in ['day', 'night']:
        for type in ['hist', 'CLAHE hist']:
            set_info['day+night'][type].extend(set_info[condition][type])
            # Theres n_images array of 4 elements, we want 4 arrays of hists. Need to transpose :)
            set_info[condition][type] = [[channel[i] for channel in set_info[condition][type]] for i in range(4)]
            # print(f"{condition} {type} shape(data)={shape(set_info[condition][type])}")
            # plotAccumulatedHist(condition, set_info[condition][type], type)
    
    condition = 'day+night'
    for type in ['hist', 'CLAHE hist']:
        set_info[condition][type] = [[channel[i] for channel in set_info[condition][type]] for i in range(4)]
        # print(f"{condition} {type} shape(data)={shape(set_info[condition][type])}")
        # plotAccumulatedHist(condition, set_info[condition][type], type)


    # PLOT BRG hist and LWIR CLAHE
    for condition in ['day', 'night', 'day+night']:
        data = [set_info[condition]['hist'][0], set_info[condition]['hist'][1], set_info[condition]['hist'][2], set_info[condition]['CLAHE hist'][3]]
        plotAccumulatedHist(condition, data, ['hist', 'hist', 'hist', 'CLAHE hist'])

    for condition in ['day', 'night', 'day+night']:
        data = [set_info[condition]['hist'][0], set_info[condition]['hist'][1], set_info[condition]['hist'][2], set_info[condition]['hist'][3]]
        plotAccumulatedHist(condition, data, 'hist')

if __name__ == '__main__':

    if not os.path.exists(os.path.join(store_path, 'pdf')):
        os.makedirs(os.path.join(store_path, 'pdf'))
    if not os.path.exists(os.path.join(store_path, 'png')):
        os.makedirs(os.path.join(store_path, 'png'))

    # Img plot for comparison
    img_path = '/home/arvc/eeha/kaist-cvpr15/images/set00/V000/lwir/I01689.jpg'
    clean_path = img_path.replace(str(home), "").replace("/eeha", "")
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    lwir = cv.calcHist([img], [0], None, [256], [0, 256])

    eq_img, clahe_hist = gethistEqCLAHE(img)
    save_images(img, eq_img, lwir, clahe_hist, 'lwir_histogram_clahe_6_6_6_comparison', clean_path)
    
    eq_img, eq_hist = gethistEq(img)
    save_images(img, eq_img, lwir, eq_hist, 'lwir_histogram_comparison', clean_path)


    print(f"Image resolution for LWIR images is: {img.shape}. Num pixels: {img.shape[0]*img.shape[1]}")
    img = cv.imread(img_path.replace("lwir", "visible"))
    print(f"Image resolution for RGB images is: {img.shape}. Num pixels: {img.shape[0]*img.shape[1]}")

    evaluateInputDataset()