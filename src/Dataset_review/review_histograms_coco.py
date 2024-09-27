
#!/usr/bin/env python3
# encoding: utf-8

"""
    Script to evaluate image historgams and the effects of the equalization on them
    on single images and in all dataset
    COCO dataset version :)
"""

import os  
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from tqdm import tqdm
from tabulate import tabulate


import pickle   

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    from utils.color_constants import c_darkgrey,c_grey,c_blue,c_green,c_yellow,c_red,c_purple
    from Dataset_review.review_dataset import home, store_path
    from Dataset_review.review_histograms import process_images, computeChannelMetrics, plot_histograms, isPathCondition, histogram_channel_cfg_list
    
coco_dataset_path = f"{home}/eeha/coco_dataset"
store_path_histogram = os.path.join(store_path, 'review_histogram', 'coco')

# Store histograms in a list of [b,g,r,lwir] hists for each image
set_info_ = {'all': {'sets': ['test2014', 'val2014', 'train2014'], 'hist': [], 'CLAHE hist': []}}   
set_info =  {'BGR': copy.deepcopy(set_info_), 'HSV': copy.deepcopy(set_info_)}  

# Test  - 40776 images
# Val   - 40504 images
# Train - 82784 images


def evaluateInputDataset():
    global set_info
    condition = 'all' # Only one condition in this case :)
    hist_type = 'hist' # No clahe here :)

    cache_file_path = os.path.join(store_path_histogram, 'set_info.pkl')
    if not os.path.exists(cache_file_path):
        with ThreadPoolExecutor() as executor:
            futures = []
            for histogram_channel_config in histogram_channel_cfg_list:
                for folder_set in os.listdir(coco_dataset_path):
                    path_set = os.path.join(coco_dataset_path, folder_set)
                    if os.path.isdir(path_set):
                        if os.path.exists(path_set):
                            args = path_set, histogram_channel_config
                            futures.append(executor.submit(process_images, args))

            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing folders'):
                hist, eq_hist, path, histogram_channel_config = future.result()
                tag = histogram_channel_config['tag']
                if isPathCondition(set_info[tag][condition]['sets'], path):
                    set_info[tag][condition]['hist'].extend(hist)
            
        with open(cache_file_path, 'wb') as f:
            pickle.dump(set_info, f)

    else:    
        print(f'Reload previous hist data stored')
        with open(cache_file_path, 'rb') as f:
            set_info = pickle.load(f)

        
    key_list = [item['tag'] for item in histogram_channel_cfg_list]
    for tag in key_list:
        # Theres n_images array of 4 elements, we want 4 arrays of hists. Need to transpose :)
        three_channel_hist_list = [[],[],[]]
        for i in range(3):
            for channel in set_info[tag][condition][hist_type]:
                three_channel_hist_list[i].append(channel[i])

        set_info[tag][condition][hist_type] = three_channel_hist_list
        print(f"[{tag}][{condition}][hist] Numer of channels: {len(set_info[tag][condition]['hist'])}; number of images: {len(set_info[tag][condition]['hist'][0])}")


    def plotAccumulatedHist(condition, data, hist_type, histogram_channel_cfg):
        hist_data = []
        
        log_table_headers = ['Test', 'CV Freq.', 'Mean F.', 'Std. F.', 'CV B.', 'Mean Bright.', 'Std B.']
        log_table_data = []
        for ch in range(3):
            log_table_data.append([f"[{condition}][{hist_type}][{histogram_channel_cfg['channel_names'][ch]}]"])
            computeChannelMetrics(data[ch], hist_data, log_table_data)

        print(tabulate(log_table_data, headers=log_table_headers, tablefmt="pretty"))

        if isinstance(hist_type, list):
            ch0_type, ch1_type, ch2_type, lwich2_type = hist_type
            hist_type = 'combined'
        else:
            ch0_type, ch1_type, ch2_type, lwich2_type = hist_type, hist_type, hist_type, hist_type

        for log_scale in [False, True]:
            plot_histograms(hist_list=[hist_data[0], hist_data[1], hist_data[2]],
                            shape=[1,1,1],
                            y_limit=histogram_channel_cfg['y_limit'],
                            labels=[f"{histogram_channel_cfg['channel_names'][0]} {ch0_type}",
                             f"{histogram_channel_cfg['channel_names'][1]} {ch1_type}",
                             f"{histogram_channel_cfg['channel_names'][2]} {ch2_type}",
                             f"NO LWIR IN COCO - {lwich2_type}"], 
                            colors=[c_blue, c_green, c_red, c_yellow],
                            filename=f'{"log_" if log_scale else ""}{condition}_coco',
                            n_images=len(data[ch]), log_scale=log_scale, condition = "", histogram_channel_cfg=histogram_channel_cfg,
                            store_path=store_path_histogram)
               
    for histogram_channel_cfg in histogram_channel_cfg_list:
        tag = histogram_channel_cfg['tag']
        plotAccumulatedHist(condition, set_info[tag][condition][hist_type], hist_type=hist_type, histogram_channel_cfg=histogram_channel_cfg)


if __name__ == '__main__':

    for hist_format in ['histogram_pdf', 'histogram_png']:
        for tag in [item['tag'] for item in histogram_channel_cfg_list]:
            os.makedirs(os.path.join(store_path_histogram, hist_format, tag), exist_ok=True)
            os.makedirs(os.path.join(store_path_histogram, hist_format, f'{tag}_log_scale'), exist_ok=True)

    evaluateInputDataset()