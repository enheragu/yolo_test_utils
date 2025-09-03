
#!/usr/bin/env python3
# encoding: utf-8

"""
    Script to evaluate image historgams and the effects of the equalization on them
    on single images and in all dataset
    LLVIP dataset version :)
"""

import os  
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from tqdm import tqdm
from tabulate import tabulate

import pickle   
import cv2 as cv

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    from utils.color_constants import c_darkgrey,c_grey,c_blue,c_green,c_yellow,c_red,c_purple
    from Dataset_review.review_dataset_llvip import home, store_path, visible_folder_name, lwir_folder_name
    from Dataset_review.review_histograms_kaist import process_images, computeChannelMetrics, plot_histograms, isPathCondition #, histogram_channel_cfg_list

histogram_channel_cfg_list = [{'tag': 'BGR', 'conversion': None, 
                               'channel_names': ['B', 'G', 'R'], 'y_limit': [19,12,19,10]}, 
                              {'tag': 'HSV', 'conversion': cv.COLOR_BGR2HSV_FULL, 
                               'channel_names': ['H', 'S', 'V'], 'y_limit': [None,None,None,None]}]

llvip_dataset_path = f"{home}/eeha/LLVIP"
store_path_histogram = os.path.join(store_path, 'review_histogram')

# Store histograms in a list of [b,g,r,lwir] hists for each image
set_info_ = {'all': {'sets': ['test', 'train'], 'hist': [], 'CLAHE hist': []}}   
set_info =  {'BGR': copy.deepcopy(set_info_), 'HSV': copy.deepcopy(set_info_)}  



def evaluateInputDataset():
    global set_info
    condition = 'all' # Only one condition in this case :)

    cache_file_path = os.path.join(store_path_histogram, 'set_info.pkl')
    if not os.path.exists(cache_file_path):
        with ThreadPoolExecutor() as executor:
            futures = []
            for histogram_channel_config in histogram_channel_cfg_list:
                for subdir in ['test', 'train']:
                    path_set = os.path.join(llvip_dataset_path, visible_folder_name, subdir)
                    if os.path.isdir(path_set) and os.path.exists(path_set):
                        args = path_set, histogram_channel_config, True
                        futures.append(executor.submit(process_images, args))

            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing folders'):
                hist, eq_hist, path, histogram_channel_config = future.result()
                tag = histogram_channel_config['tag']
                if isPathCondition(set_info[tag][condition]['sets'], path):
                    set_info[tag][condition]['hist'].extend(hist)
                    set_info[tag][condition]['CLAHE hist'].extend(eq_hist)
            
        with open(cache_file_path, 'wb') as f:
            pickle.dump(set_info, f)

    else:    
        print(f'Reload previous hist data stored')
        with open(cache_file_path, 'rb') as f:
            set_info = pickle.load(f)

        
    key_list = [item['tag'] for item in histogram_channel_cfg_list]

    for hist_type in ['hist', 'CLAHE hist']:
        for tag in key_list:
            # Theres n_images array of 4 elements, we want 4 arrays of hists. Need to transpose :)
            four_channel_hist_list = [[],[],[],[]]
            for i in range(4):
                for channel in set_info[tag][condition][hist_type]:
                    four_channel_hist_list[i].append(channel[i])

            set_info[tag][condition][hist_type] = four_channel_hist_list
            print(f"[{tag}][{condition}][hist] Numer of channels: {len(set_info[tag][condition]['hist'])}; number of images: {len(set_info[tag][condition]['hist'][0])}")


    def plotAccumulatedHist(condition, data, hist_type, histogram_channel_cfg):
        hist_data = []
        
        channel_names = histogram_channel_cfg['channel_names'] + ['LWIR']
        log_table_headers = ['Test', 'CV Freq.', 'Mean F.', 'Std. F.', 'CV B.', 'Mean Bright.', 'Std B.']
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

        for log_scale in [False, True]:
            plot_histograms(hist_list=[hist_data[0], hist_data[1], hist_data[2], hist_data[3]],
                            shape=[1,1,1],
                            y_limit=histogram_channel_cfg['y_limit'],
                            labels=[f"{histogram_channel_cfg['channel_names'][0]}", # {ch0_type}",
                             f"{histogram_channel_cfg['channel_names'][1]}", # {ch1_type}",
                             f"{histogram_channel_cfg['channel_names'][2]}", # {ch2_type}",
                             f"LWIR"], # {lwich2_type}"], 
                            colors=[c_blue, c_green, c_red, c_yellow],
                            filename=f'{hist_type.lower().replace(" ", "_")}_{"log_" if log_scale else ""}{condition}_llvip',
                            n_images=len(data[ch]), log_scale=log_scale, condition = "", histogram_channel_cfg=histogram_channel_cfg,
                            store_path=store_path_histogram)
    
    for hist_type in ['hist', 'CLAHE hist']:         
        for histogram_channel_cfg in histogram_channel_cfg_list:
            tag = histogram_channel_cfg['tag']
            plotAccumulatedHist(condition, set_info[tag][condition][hist_type], hist_type=hist_type, histogram_channel_cfg=histogram_channel_cfg)


if __name__ == '__main__':

    for hist_format in ['histogram_pdf', 'histogram_png']:
        for tag in [item['tag'] for item in histogram_channel_cfg_list]:
            os.makedirs(os.path.join(store_path_histogram, hist_format, tag), exist_ok=True)
            os.makedirs(os.path.join(store_path_histogram, hist_format, f'{tag}_log_scale'), exist_ok=True)

    evaluateInputDataset()