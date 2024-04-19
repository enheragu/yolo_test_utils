
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
import queue

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


image_queue = queue.Queue()

def save_images(img1, img2, hist1, hist2, filename):
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

def process_image(path, image_type):
    b, g, r, lwir = [], [], [], []
    eq_b, eq_g, eq_r, eq_lwir = [], [], [], []
    for file in os.listdir(path):
        if file.endswith((".jpg", ".jpeg", ".pdf", ".npy", ".npz")):
            img_path = os.path.join(path, file)
            if image_type == 'rgb':
                img = cv.imread(img_path)
                assert img is not None, f"file could not be read, check with os.path.exists() -> {file}"
                b.append(cv.calcHist([img], [0], None, [256], [0, 256]))
                g.append(cv.calcHist([img], [1], None, [256], [0, 256]))
                r.append(cv.calcHist([img], [2], None, [256], [0, 256]))
                
                b_ch,g_ch,r_ch = cv.split(img)
                eq_b.append(gethistEqCLAHE(b_ch)[1])
                eq_g.append(gethistEqCLAHE(g_ch)[1])
                eq_r.append(gethistEqCLAHE(r_ch)[1])

            elif image_type == 'lwir':
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                assert img is not None, f"file could not be read, check with os.path.exists() -> {file}"
                lwir.append(cv.calcHist([img], [0], None, [256], [0, 256]))
                
                eq_lwir.append(gethistEqCLAHE(img)[1])
                
                # Just saved once
                if save_lock.acquire(blocking=False): # not released, only to be executed once
                    # climplimig is a threshold to limit contrast. Usually between 2 and 5.
                    eq_img, clahe_hist = gethistEqCLAHE(img)
                    image_queue.put((img, eq_img, lwir[-1], clahe_hist, 'lwir_histogram_clahe_6_6_6_comparison.pdf'))
                    
                    eq_img, eq_hist = gethistEq(img)
                    image_queue.put((img, eq_img, lwir[-1], eq_hist, 'lwir_histogram_comparison.pdf'))

    return (b, g, r, lwir), (eq_b, eq_g, eq_r, eq_lwir)



def save_histogram_image(hist, title, filename, color):
    plt.figure()
    plt.plot(hist, color = color)
    # plt.fill_between(np.arange(len(hist)), hist, color=color, alpha=0.3)
    # plt.hist(hist, bins=range(len(hist)), color=color, alpha=0.7)
    plt.title(f'Histogram {title}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.close()


def plot_histograms(b_hist, g_hist, r_hist, lwir_hist, titles, colors, filename = 'averaged_histograms.pdf'):
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    for index, (hist, title, color) in enumerate(zip([b_hist, g_hist, r_hist], titles, colors)):
        
        # plt.hist(channel, bins=range(len(channel)), color=color, alpha=0.7)
        # plt.fill_between(np.arange(len(hist)), hist, color=color, alpha=0.3)
        plt.plot(hist, color = color, label = f"{title} channel")
        plt.title(f'Averaged histogram of RGB channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        save_histogram_image(hist, title, os.path.join(store_path, f'average_histogram_{title.lower()}.pdf'), color=color)
    
    plt.subplot(2, 1, 2)
    plt.title(f'Averaged histogram of LWIR channel')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(lwir_hist, color = colors[-1], label = f"{titles[-1]} channel")
    save_histogram_image(lwir_hist, titles[-1], os.path.join(store_path, f'average_histogram_{titles[-1].lower()}.pdf'), color=colors[-1])
    
    plt.tight_layout()  # Ajusta automáticamente las subfiguras para evitar solapamientos
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
                break
            break

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
    
    b_average = np.mean(b_hist, axis=0)[:,0]
    g_average = np.mean(g_hist, axis=0)[:,0]
    r_average = np.mean(r_hist, axis=0)[:,0]
    lwir_average = np.mean(lwir_hist, axis=0)[:,0]


    eq_b_average = np.mean(eq_b_hist, axis=0)[:,0]
    eq_g_average = np.mean(eq_g_hist, axis=0)[:,0]
    eq_r_average = np.mean(eq_r_hist, axis=0)[:,0]
    eq_lwir_average = np.mean(eq_lwir_hist, axis=0)[:,0]

    plot_histograms(b_average, g_average, r_average, lwir_average,
                    ["b".title(),"g".title(),"r".title(),"lwir".title()], 
                    ['#0171BA', '#78B01C', '#F23535', '#F6AE2D'],
                    f'averaged_histograms_n_{len(b_hist)}_images.pdf')
    

    plot_histograms(eq_b_average, eq_g_average, eq_r_average, eq_lwir_average,
                    ["eq_b".title(),"eq_g".title(),"eq_r".title(),"eq_lwir".title()], 
                    ['#0171BA', '#78B01C', '#F23535', '#F6AE2D'],
                    f'eq_averaged_histograms_n_{len(eq_b_hist)}_images.pdf')


if __name__ == '__main__':
    evaluateInputDataset()

    while True:
        try:
            img, eq_img, lwir_hist_orig, lwir_hist_eq, filename = image_queue.get(timeout=1)
            save_images(img, eq_img, lwir_hist_orig, lwir_hist_eq, filename)
        except queue.Empty:
            # La cola está vacía, salimos del bucle
            break