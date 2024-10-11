
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

import pickle
  
# Small hack so packages can be found
# if __name__ == "__main__":
import sys
sys.path.append('./src')
from Dataset_review.review_dataset import home, store_path
from Dataset_review.review_image_metrics import process_images, isPathCondition, computeChannelMetrics


# Store histograms in a list of [b,g,r,lwir] hists for each image
set_info = {'all': {'sets': ['test2014', 'val2014', 'train2014'], 'contrast': [], 'sharpness': []}}   
coco_dataset_path = f"{home}/eeha/coco_dataset"
store_path_imagem = os.path.join(store_path, 'image_metrics', 'coco')


def reviewImageMetrics():
    global set_info

    condition = 'all'

    cache_file_path = os.path.join(store_path_imagem, 'image_info_metrics.pkl')
    if not os.path.exists(cache_file_path):
        with ThreadPoolExecutor() as executor:
            futures = []
            for folder_set in os.listdir(coco_dataset_path):
                path_set = os.path.join(coco_dataset_path, folder_set)
                if os.path.isdir(path_set):
                    if os.path.exists(path_set):
                        futures.append(executor.submit(process_images, path_set))

            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing folders'):
                contrast, sharpness, path = future.result()
                if isPathCondition(set_info[condition]['sets'], path):
                    set_info[condition]['contrast'].extend(contrast)
                    set_info[condition]['sharpness'].extend(sharpness)
        
        with open(cache_file_path, 'wb') as f:
            pickle.dump(set_info, f)

    else:    
        print(f'Reload previous hist data stored')
        with open(cache_file_path, 'rb') as f:
            set_info = pickle.load(f)


    for data_type in ['contrast', 'sharpness']:
        ## Accumulate day+night data
        # set_info['day+night'][data_type].extend(set_info[condition][data_type])

        # Theres n_images array of 2 elements, only one channel, make it flat
        one_channel_list = []
        for channel in set_info[condition][data_type]:
            one_channel_list.append(channel[0])

        set_info[condition][data_type] = one_channel_list
 
    for data_type in ['contrast', 'sharpness']:
        data = set_info[condition][data_type]
        log_table_headers = ['Test', 'CV', 'Mean', 'Std.', 'N Img.']
        
        log_table_data = []
        log_table_data.append([f"[{condition}][{data_type}]['Visible']"])
        computeChannelMetrics(data, log_table_data)            

        print(tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty"))
        

if __name__ == '__main__':
    os.makedirs(os.path.join(store_path_imagem), exist_ok=True)
    reviewImageMetrics()
