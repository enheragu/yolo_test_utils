
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

# Output dataset whitelist (default is original Kaist sets)
white_list = ['test-all-01','test-day-20','train-all-01','train-all-20','train-day-20','train-night-20',
              'test-all-20','test-night-01','train-all-02','train-day-02','train-night-02',
              'test-day-01','test-night-20','train-all-04','train-day-04','train-night-04']



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
    
    plt.savefig(os.path.join(store_path,filename))

def gethistEqCLAHE(img):
    clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(6,6))
    eq_img = clahe.apply(img)
    clahe_hist = cv.calcHist([eq_img], [0], None, [256], [0, 256])

    return eq_img, clahe_hist

def gethistEq(img):
    eq_img = cv.equalizeHist(img) # Enhances noise too!
    hist = cv.calcHist([eq_img], [0], None, [256], [0, 256])

    return eq_img, hist


def extract_hist(img_path, img_type, plot = False):
    b, g, r, lwir = [], [], [], []
    eq_b, eq_g, eq_r, eq_lwir = [], [], [], []

    if img_type == 'rgb':
        img = cv.imread(img_path)
        assert img is not None, "file could not be read, check with os.path.exists()"
        b = [cv.calcHist([img], [0], None, [256], [0, 256])]
        g = [cv.calcHist([img], [1], None, [256], [0, 256])]
        r = [cv.calcHist([img], [2], None, [256], [0, 256])]
        
        b_ch,g_ch,r_ch = cv.split(img)
        eq_b = [gethistEqCLAHE(b_ch)[1]]
        eq_g = [gethistEqCLAHE(g_ch)[1]]
        eq_r = [gethistEqCLAHE(r_ch)[1]]

    elif img_type == 'lwir':
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        
        lwir = [cv.calcHist([img], [0], None, [256], [0, 256])]
        eq_lwir = [gethistEqCLAHE(img)[1]]
        
       
    return (b, g, r, lwir), (eq_b, eq_g, eq_r, eq_lwir)


def process_image(path, image_type):
    b, g, r, lwir = [], [], [], []
    eq_b, eq_g, eq_r, eq_lwir = [], [], [], []
    for file in os.listdir(path):
        if file.endswith((".jpg", ".jpeg", ".pdf", ".npy", ".npz")):
            img_path = os.path.join(path, file)
            ret_rgb, ret_lwir = extract_hist(img_path, image_type)

            b.extend(ret_rgb[0])
            g.extend(ret_rgb[1])
            r.extend(ret_rgb[2])
            lwir.extend(ret_rgb[3])

            eq_b.extend(ret_lwir[0])
            eq_g.extend(ret_lwir[1])
            eq_r.extend(ret_lwir[2])
            eq_lwir.extend(ret_lwir[3])

    return (b, g, r, lwir), (eq_b, eq_g, eq_r, eq_lwir)



def save_histogram_image(hist, title, filename, color, n_images, log_scale = False):
    plt.figure()
    # plt.plot(hist, color = color)
    # plt.fill_between(np.arange(len(hist)), hist, color=color, alpha=0.3)
    # plt.hist(hist, bins=range(len(hist)), color=color, alpha=0.7)

    plt.plot(hist[1], color = color, label = f"histogram {title} channel")
    # plt.plot(hist[0], color = color)
    # plt.fill_between(np.arange(len(hist[0])), hist[1], hist[0], color=color, alpha=0.3) # between min and max
    plt.fill_between(np.arange(len(hist[0])), hist[1], color=color, alpha=0.3) # between 0 and max
    if log_scale: plt.yscale('log')

    plt.title(f'{title} channel histogram ({n_images} images)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_histograms(b_hist, g_hist, r_hist, lwir_hist, titles, colors, filename = 'histograms.pdf', n_images = 0, log_scale = False):
    plt.figure(figsize=(12, 12))

    # First subplot
    plt.subplot(2, 1, 1)
    for index, (hist, title, color) in enumerate(zip([b_hist, g_hist, r_hist], titles, colors)):     
        # plt.hist(channel, bins=range(len(channel)), color=color, alpha=0.7)
        plt.plot(hist[1], color = color, label = f"{title} channel histogram")
        # plt.plot(hist[0], color = color)
        # plt.fill_between(np.arange(len(hist[0])), hist[1], hist[0], color=color, alpha=0.3) # between min and max
        plt.fill_between(np.arange(len(hist[0])), hist[1], color=color, alpha=0.3) # between 0 and max
        if log_scale: plt.yscale('log')
        save_histogram_image(hist, title, os.path.join(store_path, f'histograms_{title.lower()}_n_{n_images}_images{"_log" if log_scale else ""}.pdf'), color=color, n_images=n_images, log_scale=log_scale)
    
    plt.legend()
    plt.title(f'RGB channel histogram ({n_images} images)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(lwir_hist[1], color = colors[-1], label = f"histogram {titles[-1]} channel")
    # plt.plot(lwir_hist[0], color = colors[-1])
    # plt.fill_between(np.arange(len(lwir_hist[0])), lwir_hist[1], lwir_hist[0], color=colors[-1], alpha=0.3) # between min and max
    plt.fill_between(np.arange(len(lwir_hist[0])), lwir_hist[1], color=colors[-1], alpha=0.3) # Minimum in between 0 and max
    if log_scale: plt.yscale('log')
    save_histogram_image(lwir_hist, titles[-1], os.path.join(store_path, f'histograms_{titles[-1].lower()}_n_{n_images}_images{"_log" if log_scale else ""}.pdf'), color=colors[-1], n_images = n_images, log_scale=log_scale)
    plt.legend()
    plt.title(f'LWIR channel histogram ({n_images} images)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(store_path,filename))
    # plt.show()

set_info = {'day': {'sets': ['set00', 'set01', 'set02', 'set06', 'set07', 'set08'], 'num_img': 0, 'test': [], 'train': []},
            'night': {'sets': ['set03', 'set04', 'set05', 'set06', 'set10', 'set11'], 'num_img': 0, 'test': [], 'train': []}}

def evaluateInputDataset():
    b_hist, g_hist, r_hist, lwir_hist = [], [], [], []
    eq_b_hist, eq_g_hist, eq_r_hist, eq_lwir_hist = [], [], [], []

    with ThreadPoolExecutor() as executor:
        futures = []
        for folder_set in os.listdir(kaist_dataset_path):
            for subfolder_set in os.listdir(os.path.join(kaist_dataset_path, folder_set)):
                path_set = os.path.join(kaist_dataset_path, folder_set, subfolder_set)
                if os.path.isdir(path_set):
                    visible_folder = os.path.join(path_set, visible_folder_name)
                    if os.path.exists(visible_folder):
                        futures.append(executor.submit(process_image, visible_folder, 'rgb'))
                    
                    lwir_folder = os.path.join(path_set, lwir_folder_name)
                    if os.path.exists(lwir_folder):
                        futures.append(executor.submit(process_image, lwir_folder, 'lwir'))
                        # process_image(lwir_folder, 'lwir')
                # break
            # break

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing images'):
            (b, g, r, lwir), (eq_b, eq_g, eq_r, eq_lwir) = future.result()
            b_hist.extend(b)
            g_hist.extend(g)
            r_hist.extend(r)
            lwir_hist.extend(lwir)
            eq_b_hist.extend(eq_b)
            eq_g_hist.extend(eq_g)
            eq_r_hist.extend(eq_r)
            eq_lwir_hist.extend(eq_lwir)
    
    b_data = (np.min(b_hist, axis=0)[:,0], np.max(b_hist, axis=0)[:,0])
    g_data = (np.min(g_hist, axis=0)[:,0], np.max(g_hist, axis=0)[:,0])
    r_data = (np.min(r_hist, axis=0)[:,0], np.max(r_hist, axis=0)[:,0])
    lwir_data = (np.min(lwir_hist, axis=0)[:,0], np.max(lwir_hist, axis=0)[:,0])


    eq_b_data = (np.min(eq_b_hist, axis=0)[:,0], np.max(eq_b_hist, axis=0)[:,0])
    eq_g_data = (np.min(eq_g_hist, axis=0)[:,0], np.max(eq_g_hist, axis=0)[:,0])
    eq_r_data = (np.min(eq_r_hist, axis=0)[:,0], np.max(eq_r_hist, axis=0)[:,0])
    eq_lwir_data = (np.min(eq_lwir_hist, axis=0)[:,0], np.max(eq_lwir_hist, axis=0)[:,0])

    plot_histograms(b_data, g_data, r_data, lwir_data,
                    ["b","g","r","lwir"], 
                    ['#0171BA', '#78B01C', '#F23535', '#F6AE2D'],
                    f'histograms_all_n_{len(b_hist)}_images.pdf',
                    n_images=len(b_hist), log_scale=False)
    
    plot_histograms(b_data, g_data, r_data, lwir_data,
                    ["b","g","r","lwir"], 
                    ['#0171BA', '#78B01C', '#F23535', '#F6AE2D'],
                    f'histograms_all_n_{len(b_hist)}_images_log.pdf',
                    n_images=len(b_hist), log_scale=True)
    
    ###### Equalized

    plot_histograms(eq_b_data, eq_g_data, eq_r_data, eq_lwir_data,
                    ["clahe_b","clahe_g","clahe_r","clahe_lwir"], 
                    ['#0171BA', '#78B01C', '#F23535', '#F6AE2D'],
                    f'histograms_eq_all_n_{len(eq_b_hist)}_images.pdf',
                    n_images=len(eq_b_hist), log_scale=False)

    plot_histograms(eq_b_data, eq_g_data, eq_r_data, eq_lwir_data,
                    ["clahe_b","clahe_g","clahe_r","clahe_lwir"], 
                    ['#0171BA', '#78B01C', '#F23535', '#F6AE2D'],
                    f'histograms_eq_all_n_{len(eq_b_hist)}_images_log.pdf',
                    n_images=len(eq_b_hist), log_scale=True)

if __name__ == '__main__':

    # Img plot for comparison
    img_path = '/home/arvc/eeha/kaist-cvpr15/images/set00/V000/lwir/I01689.jpg'
    clean_path = img_path.replace(str(home), "").replace("/eeha", "")
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    lwir = cv.calcHist([img], [0], None, [256], [0, 256])

    eq_img, clahe_hist = gethistEqCLAHE(img)
    save_images(img, eq_img, lwir, clahe_hist, 'lwir_histogram_clahe_6_6_6_comparison.pdf', clean_path)
    
    eq_img, eq_hist = gethistEq(img)
    save_images(img, eq_img, lwir, eq_hist, 'lwir_histogram_comparison.pdf', clean_path)


    print(f"Image resolution for LWIR images is: {img.shape}. Num pixels: {img.shape[0]*img.shape[1]}")
    img = cv.imread(img_path.replace("lwir", "visible"))
    print(f"Image resolution for LWIR images is: {img.shape}. Num pixels: {img.shape[0]*img.shape[1]}")

    evaluateInputDataset()