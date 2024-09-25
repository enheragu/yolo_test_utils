
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
from tabulate import tabulate

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
from Dataset.static_image_compression import combine_hsvt, combine_rgbt, combine_vt, combine_vths
from Dataset.pca_fa_compression import combine_rgbt_fa_to3ch, combine_rgbt_pca_to3ch
from utils import color_palette_list
from utils.color_constants import c_darkgrey,c_grey,c_blue,c_green,c_yellow,c_red,c_purple
from Dataset_review.review_dataset import home, kaist_dataset_path, kaist_yolo_dataset_path, labels_folder_name, images_folder_name, visible_folder_name, lwir_folder_name, store_path


histogram_channel_cfg_list = [{'tag': 'BGR', 'conversion': None, 'channel_names': ['B', 'G', 'R']},
                              {'tag': 'HSV', 'conversion': cv.COLOR_BGR2HSV_FULL, 'channel_names': ['H', 'S', 'V']}]

# Store histograms in a list of [b,g,r,lwir] hists for each image
set_info_ = {'day': {'sets': ['set00', 'set01', 'set02', 'set06', 'set07', 'set08'], 'hist': [], 'CLAHE hist': []},
            'night': {'sets': ['set03', 'set04', 'set05', 'set09', 'set10', 'set11'], 'hist': [], 'CLAHE hist': []},
            'day+night': {'hist': [], 'CLAHE hist': []}}
            
set_info =  {'BGR': copy.deepcopy(set_info_), 'HSV': copy.deepcopy(set_info_)}    


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
    plt.savefig(os.path.join(store_path,'histogram_pdf',f'{filename}.pdf'))
    plt.savefig(os.path.join(store_path,'histogram_png',f'{filename}.png'))
    plt.close()

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

def extract_hist(img_path, histogram_channel_config, plot = False):
    ch0, ch1, ch2, lwir = [], [], [], []
    eq_ch0, eq_ch1, eq_ch2, eq_lwir = [], [], [], []

    # Process visible image
    img = readImage(img_path)
    assert img is not None, "file could not be read, check with os.path.exists()"

    if histogram_channel_config['conversion'] is not None:
        image = cv.cvtColor(img, histogram_channel_config['conversion'])
    else:
        image = img
        
    ch0 = cv.calcHist([image], [0], None, [256], [0, 256])
    ch1 = cv.calcHist([image], [1], None, [256], [0, 256])
    ch2 = cv.calcHist([image], [2], None, [256], [0, 256])
    
    ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0], _ = gethistEqCLAHE(ycrcb_img[:, :, 0]) 
    equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)

    if histogram_channel_config['conversion'] is not None:
        equalized_image = cv.cvtColor(equalized_img, histogram_channel_config['conversion'])
    else:
        equalized_image = equalized_img
        
    eq_ch0 = cv.calcHist([equalized_image], [0], None, [256], [0, 256])
    eq_ch1 = cv.calcHist([equalized_image], [1], None, [256], [0, 256])
    eq_ch2 = cv.calcHist([equalized_image], [2], None, [256], [0, 256])

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
        
    return [ch0, ch1, ch2, lwir], [eq_ch0, eq_ch1, eq_ch2, eq_lwir]


def process_images(args):
    path, histogram_channel_config = args
    hist = []
    eq_hist = []
    for file in os.listdir(path):
        if file.endswith((".jpg", ".jpeg", ".pdf", ".npy", ".npz")):
            img_path = os.path.join(path, file)
            ret_hist, ret_eq_hist = extract_hist(img_path, histogram_channel_config)

            hist.append(ret_hist)
            eq_hist.append(ret_eq_hist)

    return hist, eq_hist, path, histogram_channel_config


def plotHistogramHeatmap(data, ax, y_bin_num = 30, log_scale = False):
    # List of histograms
    histogram_array = np.squeeze(np.array(data))
    num_images, grey_scale = histogram_array.shape
    
    # "#ffffff", ... ,c_grey
    color_palette = [c_purple,c_blue,c_green,c_yellow,c_red,c_darkgrey]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_palette, N=y_bin_num)
    # cmap = 'RdYlGn_r'
    # cmap = 'inferno'
    # cmap = plt.get_cmap('Spectral', y_bin_num)
    vmin=1
    vmax = np.max(histogram_array) + 1e-10
    min_freq = 0
    
    try:
        max_density = vmin
        for idx in range(grey_scale):
            density_array, _ = np.histogram(histogram_array[:,idx], bins=y_bin_num)
            max_density = max(np.max(density_array), max_density)

        for idx in range(grey_scale):
            max_freq = np.max(histogram_array[:,idx])
            density_array, _ = np.histogram(histogram_array[:,idx], bins=y_bin_num)
            if np.any(density_array == 0): # If all zero -> Zero to small amount :)
                density_array[density_array == 0] = 1e-10
            # Generate heatmap to this bin
            density_array_vertical = density_array.reshape(-1, 1)
            # density_array_vertical[density_array_vertical<max_density*0.002] = 0

            extra_x = 0 if not log_scale else 0.2
            if max_freq > 0:
                ax.imshow(density_array_vertical, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=max_density),
                        extent=[idx, idx+1+extra_x, min_freq, max_freq],
                        origin='lower', aspect='auto')
            
        ax.autoscale()

        cb = plt.colorbar(ax.images[0], ax=ax, label='Num Images')
        tick_locator = LogLocator(numticks=y_bin_num)
        cb.locator = tick_locator
        cb.update_ticks()
    
    except Exception as e:
        print(f'Catched expection: {e}')
        raise e
    
    
def save_histogram_image(hist, title, filename, color, n_images, log_scale = False, condition = '-', histogram_channel_cfg={}, store_path = store_path):
    # plt.figure(figsize=(4.5, 3))
    fig, ax = plt.subplots(figsize=(9, 6))

    # ax.plot(hist[1], color = color, label = f"{title} (max)") # Plots max of histograms    
    # plt.fill_between(np.arange(len(hist['min'])), hist['min'], color=color, alpha=0.8, label = "min") # between 0 and min

    plotHistogramHeatmap(data = hist['data'], ax = ax, log_scale=log_scale)

    if log_scale: 
        plt.yscale('log') # y log scale is handled when generating  imshow for a better colorbar fitting
        plt.minorticks_on()

    plt.title(f'{title} histogram ({n_images} images) {condition}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency' + (' (log scale)' if log_scale else ''))
    
    plt.tight_layout()
    plt.savefig(os.path.join(store_path,'histogram_pdf',histogram_channel_cfg['tag'],f'{filename}.pdf'))
    plt.savefig(os.path.join(store_path,'histogram_png',histogram_channel_cfg['tag'],f'{filename}.png'))
    plt.close()


def plot_histograms(b_hist, g_hist, r_hist, lwir_hist, titles, colors, filename = 'histograms', n_images = 0, log_scale = False, condition = '-', histogram_channel_cfg={}, store_path=store_path):
    plt.figure(figsize=(9, 3))

    # First subplot
    if lwir_hist is not None:
        ax = plt.subplot(1, 2, 1)
    else:
        ax = plt.subplot(1,1,1)

    for hist, title, color in zip([b_hist, g_hist, r_hist], titles, colors):
        if hist is None: 
            continue
        ax.plot(hist['average'], color = color, label = f"{title}") # Plots max of histograms
        ax.plot(hist['max'], color = color, linestyle='dashed', linewidth=0.5)
        ax.plot(hist['min'], color = color, linestyle='dashed', linewidth=0.5)
        # plt.fill_between(np.arange(len(hist['min'])), hist['min'], color=color, alpha=0.8, label = "min histogram") # between 0 and min
        # ax.fill_between(np.arange(len(hist['min'])), hist['max'], color=color, alpha=0.3) #, label = "variance") # between 0 and max

        if log_scale: 
            plt.yscale('log')
            plt.minorticks_on()
        # Single histogram for each BGR channel
        save_histogram_image(hist, title, f'{title.lower().replace(" ", "_")}_{filename}', color=color, n_images=n_images, log_scale=log_scale, condition=condition, histogram_channel_cfg=histogram_channel_cfg, store_path=store_path)
    
    plt.legend()
    plt.title(f'RGB histograms ({n_images} images) {condition}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency' + (' (log scale)' if log_scale else ''))

    if lwir_hist is not None:
        # Second subplot
        ax = plt.subplot(1, 2, 2)
        ax.plot(lwir_hist['average'], color = colors[-1], label = f"{titles[-1]}")
        ax.plot(lwir_hist['max'], color = colors[-1], linestyle='dashed', linewidth=0.5)
        ax.plot(lwir_hist['min'], color = colors[-1], linestyle='dashed', linewidth=0.5)
        # plt.fill_between(np.arange(len(lwir_hist['min'])), lwir_hist['min'], color=colors[-1], alpha=0.8, label = "min histogram") # between 0 and min
        # ax.fill_between(np.arange(len(lwir_hist['min'])), lwir_hist['max'], color=colors[-1], alpha=0.3) #, label = "variance") # Minimum in between 0 and max    
        

        if log_scale: 
            plt.yscale('log')
            plt.minorticks_on()
        save_histogram_image(lwir_hist, titles[-1], f'{titles[-1].lower().replace(" ", "_")}_{filename}', color=colors[-1], n_images = n_images, log_scale=log_scale, condition=condition, histogram_channel_cfg=histogram_channel_cfg, store_path=store_path)
        plt.legend()
        plt.title(f'LWIR histogram ({n_images} images) {condition}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency' + (' (log scale)' if log_scale else ''))

    plt.tight_layout()
    plt.savefig(os.path.join(store_path,'histogram_pdf',histogram_channel_cfg['tag'],f'{filename}.pdf'))
    plt.savefig(os.path.join(store_path,'histogram_png',histogram_channel_cfg['tag'],f'{filename}.png'))
    plt.close()
    # plt.show()


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

def computeHistogramsModes(data_ch, y_bin_num = 30):
    modes = []
    
    density_matrix = (np.array(data_ch).T)[0]
    num_bins, num_images = density_matrix.shape
    
    # print(f"{density_matrix.shape = }")
    vmax = np.max(density_matrix)
    
    for idx in range(num_bins):
        column = density_matrix[idx,:]
        hist_values, hist_edges = np.histogram(column, bins=np.linspace(0, vmax, y_bin_num))

        mode_bin_index = np.argmax(hist_values) # higher freq. bin :)
        mode_value = (vmax/y_bin_num)*(mode_bin_index+0.5)
        # print(f"{y_bin_num = }; {mode_bin = }; {mode_value = }; {vmax = }")
        modes.append(mode_value)
        
    return np.array(modes)

def computeChannelMetrics(datach,hist_data,log_table):
    data_ch = np.array(datach)
    min_vals = np.min(data_ch, axis=0)
    max_vals = np.max(data_ch, axis=0)
    mean_vals = np.mean(data_ch, axis=0)  

    std_vals = np.std(data_ch, axis=0)
    max_vals = mean_vals + 4*std_vals
    min_vals = np.maximum(mean_vals - 4 * std_vals, 0)

    mean_mean = np.mean(mean_vals)
    std_mean = np.mean(std_vals)
    variation_coef =  (std_mean / mean_mean) * 100

    brightness_values = []

    # print(f"{data_ch.shape}")
    for histogram in (data_ch.squeeze(axis=2)):
        normalized_histogram = histogram / np.sum(histogram)
        brightness = np.sum(normalized_histogram * np.arange(256))
        brightness_values.append(brightness)

    brightness_values = np.array(brightness_values)
    mean_brightness = np.mean(brightness_values)
    std_brightness = np.std(brightness_values)
    variation_coef_brightness =  (std_brightness / mean_brightness) * 100
                
    hist_data.append({'min': min_vals, 'max': max_vals, 'average': mean_vals, 'data': datach,
                        'std': std_vals, 'mean_stds': std_mean, 'mean_means': mean_mean, 'cv': variation_coef,
                        'mean_brightness': mean_brightness, 'std_brightness':std_brightness, 'cv_brightness': variation_coef_brightness})
    
    if log_table is not None:
        log_table[-1].extend([f"{variation_coef:.3f}",
                              f"{mean_mean:.2f}",
                              f"{std_mean:.3f}",
                              f"{mean_brightness:.3f}",
                              f"{std_brightness:.3f}",
                              f"{variation_coef_brightness:.3f}"])

    ## ¿Intervals that could be considered?
    # Dark image: Average brightness between 0 and 85.
    # Normal image: Average brightness between 85 and 170.
    # Bright image: Average brightness between 170 and 255.

def evaluateInputDataset():
    global set_info

    cache_file_path = os.path.join(store_path, 'set_info.pkl')
    if not os.path.exists(cache_file_path):
        with ThreadPoolExecutor() as executor:
            futures = []
            for histogram_channel_config in histogram_channel_cfg_list:
                for folder_set in os.listdir(kaist_dataset_path):
                    for subfolder_set in os.listdir(os.path.join(kaist_dataset_path, folder_set)):
                        path_set = os.path.join(kaist_dataset_path, folder_set, subfolder_set)
                        if os.path.isdir(path_set):
                            visible_folder = os.path.join(path_set, visible_folder_name)
                            if os.path.exists(visible_folder):
                                args = visible_folder, histogram_channel_config
                                futures.append(executor.submit(process_images, args))
                #     break
                # break

            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing folders'):
                hist, eq_hist, path, histogram_channel_config = future.result()
                tag = histogram_channel_config['tag']
                for condition in ['day', 'night']:
                    if isPathCondition(set_info[tag][condition]['sets'], path):
                        set_info[tag][condition]['hist'].extend(hist)
                        set_info[tag][condition]['CLAHE hist'].extend(eq_hist)
                        break
        
        with open(cache_file_path, 'wb') as f:
            pickle.dump(set_info, f)

    else:    
        print(f'Reload previous hist data stored')
        with open(cache_file_path, 'rb') as f:
            set_info = pickle.load(f)


    # This is done once loaded to not interfere with the heavy process before in case of any issue
    # day+night is just the sum of previous, so is not worth storing duplicated data in the pickle file
    key_list = [item['tag'] for item in histogram_channel_cfg_list]
    for tag in key_list:
        for condition in ['day', 'night']:
            for hist_type in ['hist', 'CLAHE hist']:
                ## Accumulate day+night data
                # set_info[tag]['day+night'][hist_type].extend(set_info[tag][condition][hist_type])

                # Theres n_images array of 4 elements, we want 4 arrays of hists. Need to transpose :)
                four_channel_hist_list = [[],[],[],[]]
                for i in range(4):
                    for channel in set_info[tag][condition][hist_type]:
                        four_channel_hist_list[i].append(channel[i])

                set_info[tag][condition][hist_type] = four_channel_hist_list
 
        # condition = 'day+night'
        # for hist_type in ['hist', 'CLAHE hist']:four_channel_hist_list = [[],[],[],[]]
        #     for i in range(4):
        #         for channel in set_info[tag][condition][hist_type]:
        #             four_channel_hist_list[i].append(channel[i])

        #     set_info[tag][condition][hist_type] = four_channel_hist_list


    def plotAccumulatedHist(condition, data, hist_type, histogram_channel_cfg):
        hist_data = []
        channel_names = histogram_channel_cfg['channel_names'] + ['LWIR']

        log_table_headers = ['Test', 'CV Freq.', 'Mean F.', 'Std. F.', 'Mean Bright.', 'Std B.', 'CV B.']
        log_table_data = []
        for ch in range(4):
            log_table_data.append([f"[{condition}][{hist_type}][{channel_names[ch]}]"])
            computeChannelMetrics(data[ch], hist_data, log_table_data)            

        print(tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty"))

        if isinstance(hist_type, list):
            ch0_type, ch1_type, ch2_type, lwich2_type = hist_type
            hist_type = 'combined'
        else:
            ch0_type, ch1_type, ch2_type, lwich2_type = hist_type, hist_type, hist_type, hist_type

        for log_scale in [True, False]:
            plot_histograms(hist_data[0], hist_data[1], hist_data[2], hist_data[3],
                            [f"{histogram_channel_cfg['channel_names'][0]} {ch0_type}",
                             f"{histogram_channel_cfg['channel_names'][1]} {ch1_type}",
                             f"{histogram_channel_cfg['channel_names'][2]} {ch2_type}",
                             f"LWIR {lwich2_type}"], 
                            [c_blue, c_green, c_red, c_yellow],
                            f'{"log_" if log_scale else ""}{condition}_{hist_type}',
                            n_images=len(data[ch]), log_scale=log_scale, condition = f"({condition} condition)", histogram_channel_cfg=histogram_channel_cfg,
                            store_path=store_path)

            
    for hist_type in ['CLAHE hist', 'hist']:
        for histogram_channel_cfg in histogram_channel_cfg_list:
            tag = histogram_channel_cfg['tag']
            for condition in ['day', 'night']: #, 'day+night']:
                plotAccumulatedHist(condition, set_info[tag][condition][hist_type], hist_type=hist_type, histogram_channel_cfg=histogram_channel_cfg) #['hist', 'hist', 'hist', 'CLAHE hist'])



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

    if not os.path.exists(os.path.join(store_path, 'histogram_pdf')):
        os.makedirs(os.path.join(store_path, 'histogram_pdf'))
        for tag in [item['tag'] for item in histogram_channel_cfg_list]:
            os.makedirs(os.path.join(store_path, 'histogram_pdf', tag))
    if not os.path.exists(os.path.join(store_path, 'histogram_png')):
        os.makedirs(os.path.join(store_path, 'histogram_png'))
        for tag in [item['tag'] for item in histogram_channel_cfg_list]:
            os.makedirs(os.path.join(store_path, 'histogram_png', tag))

    # evaluateEqualizationMethods()

    evaluateInputDataset()