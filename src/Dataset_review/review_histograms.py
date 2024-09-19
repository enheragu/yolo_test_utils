
#!/usr/bin/env python3
# encoding: utf-8

"""
    Script to evaluate image historgams and the effects of the equalization on them
    on single images and in all dataset
"""

import os  
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
import copy
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm 
from matplotlib.ticker import LogLocator
from matplotlib.colors import LinearSegmentedColormap

import pickle
import cv2 as cv


# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    from Dataset.static_image_compression import combine_hsvt, combine_rgbt, combine_vt, combine_vths
    from Dataset.pca_fa_compression import combine_rgbt_fa_to3ch, combine_rgbt_pca_to3ch
    from utils import color_palette_list
    from utils.color_constants import c_darkgrey,c_grey,c_blue,c_green,c_yellow,c_red,c_purple
    from Dataset_review.review_dataset import home, kaist_dataset_path, kaist_yolo_dataset_path, labels_folder_name, images_folder_name, visible_folder_name, lwir_folder_name, store_path

def readImage(path, flag = None):
    if flag is not None:
        imagen = cv.imread(path, flag)
    else:
        imagen = cv.imread(path)
    img_heigth, img_width = imagen.shape[:2]
    aspect_ratio = img_width / img_heigth
    cut_pixels_width = 2
    cut_pixels_height = int(cut_pixels_width/aspect_ratio)
    imagen_recortada = imagen[cut_pixels_height:img_heigth-cut_pixels_height, cut_pixels_width:img_width-cut_pixels_width]
    imagen_final = cv.resize(imagen_recortada, (img_width, img_heigth))
    
    return imagen_final

def save_images(img_hist_list, filename, img_path):
    rows = 2
    cols = len(img_hist_list)

    item = 1
    plt.figure(figsize=[cols*6.4, rows*5.12])
    for row_daat in img_hist_list:
        plt.subplot(rows, cols, item)
        plt.imshow(row_daat['img'])
        plt.title(row_daat['img_title'])
        plt.axis('off')

        plt.subplot(rows, cols, item+cols)
        plt.plot(row_daat['hist'], color = '#F6AE2D')
        plt.title(row_daat['hist_title'])
        # plt.axis('off')
        item+=1   
    
    plt.annotate(f'Img: {img_path}',
                    xy = (1.0, -0.17), xycoords='axes fraction',
                    ha='right', va="center", fontsize=12,
                    color='black', alpha=0.2)
    
    plt.tight_layout()
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
    hist, bins = np.histogram(img.flatten(), 256, [0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf_normalized, 0) # Avoid zero division
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    expanded_hist = cv.calcHist([cdf[img]], [0], None, [256], [0, 256])
    return cdf[img], expanded_hist # Apply transformation to original image

def extract_hist(img_path, plot = False):
    b, g, r, lwir = [], [], [], []
    eq_b, eq_g, eq_r, eq_lwir = [], [], [], []

    # Process visible image
    img = readImage(img_path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    b = cv.calcHist([img], [0], None, [256], [0, 256])
    g = cv.calcHist([img], [1], None, [256], [0, 256])
    r = cv.calcHist([img], [2], None, [256], [0, 256])
    
    ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0], _ = gethistEqCLAHE(ycrcb_img[:, :, 0]) 
    equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)

    eq_b = cv.calcHist([equalized_img], [0], None, [256], [0, 256])
    eq_g = cv.calcHist([equalized_img], [1], None, [256], [0, 256])
    eq_r = cv.calcHist([equalized_img], [2], None, [256], [0, 256])

    # b_ch,g_ch,r_ch = cv.split(img)
    # eq_b = gethistEqCLAHE(b_ch)[1]
    # eq_g = gethistEqCLAHE(g_ch)[1]
    # eq_r = gethistEqCLAHE(r_ch)[1]

    # Process LWIR correspondant image
    img_path = img_path.replace(visible_folder_name, lwir_folder_name)
    img = readImage(img_path, cv.IMREAD_GRAYSCALE)
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


def plotHistogramHeatmap(data, ax, y_bin_num = 30):

    density_matrix = (np.array(data).T)[0]
    num_bins, num_images = density_matrix.shape
    
    # ,c_grey
    color_palette = [c_purple,c_blue,c_green,c_yellow,c_red,c_darkgrey]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_palette, N=y_bin_num)
    # cmap = 'RdYlGn_r'
    # cmap = 'inferno'
    # cmap = plt.get_cmap('Spectral', y_bin_num)
    vmin=1
    max_frequency = 0

    def compute_hist_log_space(column):
        non_zero_column = column[column > 0]  # Exclude zeros from the column for geomspace
        if len(non_zero_column) == 0:
            return None

        hist_values, hist_edges = np.histogram(non_zero_column, bins=np.geomspace(1, np.max(non_zero_column), y_bin_num)) # np.arange(y_bin_num))
        binned_col = hist_values.reshape(-1, 1)
        return binned_col
    
    try:
        for idx in range(num_bins):
            binned_col = compute_hist_log_space(density_matrix[idx,:])
            if binned_col is None:
                continue
            max_frequency = max(max_frequency, np.max(binned_col))
        
        vmax = max_frequency
        
        for idx in range(num_bins):
            column = density_matrix[idx,:]
            bin_height = int(np.ceil(np.max(column)))
            bin_floor = int(np.floor(np.min(column)))
                        
            binned_col = compute_hist_log_space(density_matrix[idx,:])
            if binned_col is None:
                continue
            
            # Generate heatmap to this bin
            ax.imshow(binned_col, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), # vmin=vmin, vmax=vmax,
                    extent=[idx, idx+1, bin_floor, bin_height],
                    origin='lower', aspect='auto')
            
            # Adds border to each heatmap
            # ax.add_patch(mpatches.Rectangle((idx, 0), 1, bin_height, fc='none', ec='k', lw=1))
        
        ax.autoscale()

        cb = plt.colorbar(ax.images[0], ax=ax, label='Num Images')
        tick_locator = LogLocator(numticks=y_bin_num)
        cb.locator = tick_locator
        cb.update_ticks()
    
    except Exception as e:
        print(f'Catched expection: {e}')
    
    

def save_histogram_image(hist, title, filename, color, n_images, log_scale = False, condition = '-'):
    # plt.figure(figsize=(4.5, 3))
    fig, ax = plt.subplots(figsize=(9, 6))

    # ax.plot(hist[1], color = color, label = f"{title} (max)") # Plots max of histograms    
    # plt.fill_between(np.arange(len(hist[0])), hist[0], color=color, alpha=0.8, label = "min") # between 0 and min

    if True: #not log_scale:
        # ax.plot(hist[0], color = color)                         # Plots min of histograms
        plotHistogramHeatmap(data = hist[2], ax = ax)
        # ax.invert_yaxis()
    else:
        ax.fill_between(np.arange(len(hist[0])), hist[1], color=color, alpha=0.3) #, label = "variance") # between 0 and max


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
    ax = plt.subplot(1, 2, 1)
    for hist, title, color in zip([b_hist, g_hist, r_hist], titles, colors):
        ax.plot(hist[1], color = color, label = f"{title} (max)") # Plots max of histograms
        # plt.fill_between(np.arange(len(hist[0])), hist[0], color=color, alpha=0.8, label = "min histogram") # between 0 and min
        # ax.fill_between(np.arange(len(hist[0])), hist[1], color=color, alpha=0.3) #, label = "variance") # between 0 and max

        if log_scale: plt.yscale('log')
        # Single histogram for each BGR channel
        save_histogram_image(hist, title, f'{title.lower().replace(" ", "_")}_{filename}', color=color, n_images=n_images, log_scale=log_scale)
    
    plt.legend()
    plt.title(f'RGB histograms ({n_images} images) ({condition})')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency' + (' (log scale)' if log_scale else ''))

    # Second subplot
    ax = plt.subplot(1, 2, 2)
    ax.plot(lwir_hist[1], color = colors[-1], label = f"{titles[-1]} (max)")
    # plt.fill_between(np.arange(len(lwir_hist[0])), lwir_hist[0], color=colors[-1], alpha=0.8, label = "min histogram") # between 0 and min
    # ax.fill_between(np.arange(len(lwir_hist[0])), lwir_hist[1], color=colors[-1], alpha=0.3) #, label = "variance") # Minimum in between 0 and max    
      

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
    global set_info

    cache_file_path = os.path.join(store_path, 'set_info.pkl')
    if not os.path.exists(cache_file_path):
        with ThreadPoolExecutor() as executor:
            futures = []
            for folder_set in os.listdir(kaist_dataset_path):
                for subfolder_set in os.listdir(os.path.join(kaist_dataset_path, folder_set)):
                    path_set = os.path.join(kaist_dataset_path, folder_set, subfolder_set)
                    if os.path.isdir(path_set):
                        visible_folder = os.path.join(path_set, visible_folder_name)
                        if os.path.exists(visible_folder):
                            futures.append(executor.submit(process_image, visible_folder))
                #     break
                # break

            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing imageSet folders'):
                hist, eq_hist, path = future.result()
                for condition in ['day', 'night']:
                    if isPathCondition(condition, path):
                        set_info[condition]['hist'].extend(hist)
                        set_info[condition]['CLAHE hist'].extend(eq_hist)
                        break
                
        for condition in ['day', 'night']:
            for type in ['hist', 'CLAHE hist']:
                set_info['day+night'][type].extend(set_info[condition][type])
                # Theres n_images array of 4 elements, we want 4 arrays of hists. Need to transpose :)
                set_info[condition][type] = [[channel[i] for channel in set_info[condition][type]] for i in range(4)]
                # print(f"{condition} {type} shape(data)={shape(set_info[condition][type])}")
                # plotAccumulatedHist(condition, set_info[condition][type], type)
        
        
        with open(cache_file_path, 'wb') as f:
            pickle.dump(set_info, f)

    else:    
        print(f'Reload previous hist data stored')
        with open(cache_file_path, 'rb') as f:
            set_info = pickle.load(f)

        
    def plotAccumulatedHist(condition, data, type):
        hist_data = []
        for ch in range(4):
            min_vals = np.min(data[ch], axis=0)[:,0]
            max_vals = np.max(data[ch], axis=0)[:,0]
            hist_data.append([min_vals, max_vals, data[ch]])
        
        if isinstance(type, list):
            b_type, g_type, r_type, lwir_type = type
            type = 'combined'
        else:
            b_type, g_type, r_type, lwir_type = type, type, type, type


        for log_scale in [True, False]:
            plot_histograms(hist_data[0], hist_data[1], hist_data[2], hist_data[3],
                            [f"B {b_type}",f"G {g_type}",f"R {r_type}",f"LWIR {lwir_type}"], 
                            [c_blue, c_green, c_red, c_yellow],
                            f'{"log_" if log_scale else ""}{condition}_{type}',
                            n_images=len(data[ch]), log_scale=log_scale, condition = condition)
                
    condition = 'day+night'
    for type in ['hist', 'CLAHE hist']:
        set_info[condition][type] = [[channel[i] for channel in set_info[condition][type]] for i in range(4)]
        # print(f"{condition} {type} shape(data)={shape(set_info[condition][type])}")
        # plotAccumulatedHist(condition, set_info[condition][type], type)

    # PLOT BRG hist and LWIR CLAHE
    for condition in ['day', 'night']: #, 'day+night']:
        data = [set_info[condition]['hist'][0], set_info[condition]['hist'][1], set_info[condition]['hist'][2], set_info[condition]['CLAHE hist'][3]]
        plotAccumulatedHist(condition, data, 'CLAHE hist') #['hist', 'hist', 'hist', 'CLAHE hist'])

    for condition in ['day', 'night']: #, 'day+night']:
        data = [set_info[condition]['hist'][0], set_info[condition]['hist'][1], set_info[condition]['hist'][2], set_info[condition]['hist'][3]]
        plotAccumulatedHist(condition, data, 'hist')


def evaluateEqualizationMethods():
    # Img plot for comparison
    img_path = '/home/arvc/eeha/kaist-cvpr15/images/set00/V000/lwir/I01689.jpg'
    clean_path = img_path.replace(str(home), "").replace("/eeha", "")
    img = readImage(img_path, cv.IMREAD_GRAYSCALE)
    lwir = cv.calcHist([img], [0], None, [256], [0, 256])

    org_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    org_lwir = cv.calcHist([img], [0], None, [256], [0, 256])

    plot = [{'img':img,'img_title': 'Cut Image','hist':lwir,'hist_title': 'Cut Image Hist'},
            {'img':org_img,'img_title': 'Original Histogram Image','hist':org_lwir,'hist_title': 'Original Image Hist'}]
    save_images(plot,'lwir_histogram_cut_comparison', clean_path)


    eq_img, clahe_hist = gethistEqCLAHE(img)
    plot = [{'img':org_img,'img_title': 'Original Image','hist':lwir,'hist_title': 'Original Image Hist'},
            {'img':eq_img,'img_title': 'CLAHE-Histogram Image','hist':clahe_hist,'hist_title': 'CLAHE Histogram'}]
    save_images(plot,'lwir_histogram_clahe_6_6_6_comparison', clean_path)


    
    eq_img, eq_hist = gethistEq(img)
    plot = [{'img':img,'img_title': 'Original Image','hist':lwir,'hist_title': 'Original Image Hist'},
            {'img':eq_img,'img_title': 'Eq-Histogram Image','hist':eq_hist,'hist_title': 'Eq Histogram'}]
    save_images(plot,'lwir_histogram_equalization_comparison', clean_path)

    exp_img, exp_hist = gethistExpLinear(img)
    plot = [{'img':img,'img_title': 'Original Image','hist':lwir,'hist_title': 'Original Image Hist'},
            {'img':exp_img,'img_title': 'Eq-Histogram Image','hist':exp_hist,'hist_title': 'Eq Histogram'}]
    save_images(plot,'lwir_histogram_expanded_comparison', clean_path)



    
    plot = [{'img':img,'img_title': 'Original Image','hist':lwir,'hist_title': 'Original Image Hist'},
            {'img':eq_img,'img_title': 'Eq-Histogram Image','hist':eq_hist,'hist_title': 'Eq Histogram'},
            {'img':eq_img,'img_title': 'CLAHE-Histogram Image','hist':clahe_hist,'hist_title': 'CLAHE Histogram'}]
    save_images(plot,'lwir_histogram_all_comparison', clean_path)


    print(f"Image resolution for LWIR images is: {img.shape}. Num pixels: {img.shape[0]*img.shape[1]}")
    img = readImage(img_path.replace("lwir", "visible"))
    print(f"Image resolution for RGB images is: {img.shape}. Num pixels: {img.shape[0]*img.shape[1]}")



if __name__ == '__main__':

    if not os.path.exists(os.path.join(store_path, 'pdf')):
        os.makedirs(os.path.join(store_path, 'pdf'))
    if not os.path.exists(os.path.join(store_path, 'png')):
        os.makedirs(os.path.join(store_path, 'png'))
    
    evaluateEqualizationMethods()

    evaluateInputDataset()