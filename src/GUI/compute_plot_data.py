#!/usr/bin/env python3
# encoding: utf-8
"""
    Based on YAML data computes the plot information (interpolation of curves, compute metrics, etc)
"""


import math
import numpy as np
import inspect

from utils import log, bcolors

# Extracted from https://github.com/Cartucho/mAP/blob/3605865a350859e60c7b711838d09c4e0012c774/main.py#L80
def log_average_miss_rate(mr, fppi):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
            State of the Art." Pattern Analysis and Machine Intelligence, IEEE
            Transactions on 34.4 (2012): 743 - 761.
    """
    # if there were no detections of that class
    if mr.size == 0:
        return 0

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    result = np.zeros(ref.shape)
    for i, ref_i in enumerate(ref):
        # will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.argwhere(fppi_tmp <= ref_i).flatten()
        if j.size:
            result[i] = mr_tmp[j[-1]]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr


# Version from https://github.com/Tencent/ObjectDetection-OneStageDet/blob/master/brambox/boxes/statistics/mr_fppi.py
def compute_lamr(miss_rate, fppi, num_of_samples=9):
    import scipy.interpolate  
    """ Compute the log average miss-rate from a given MR-FPPI curve.
    The log average miss-rate is defined as the average of a number of evenly spaced log miss-rate samples
    on the :math:`{log}(FPPI)` axis within the range :math:`[10^{-2}, 10^{0}]`

    Args:
        miss_rate (list): miss-rate values
        fppi (list): FPPI values
        num_of_samples (int, optional): Number of samples to take from the curve to measure the average precision; Default **9**

    Returns:
        Number: log average miss-rate
    """
    samples = np.logspace(-2., 0., num_of_samples)
    interpolated = scipy.interpolate.interp1d(fppi, miss_rate, fill_value=(1., 0.), bounds_error=False)(samples)
    log_interpolated = np.log(np.maximum(1e-10, interpolated))
    lamr = np.exp(np.mean(log_interpolated))

    return lamr


def getBestTag(data):
    ## FILTER UNUSED DATA TO AVOID MEMORY CONSUMPTION
    last_fit_tag = 'pr_data_' + str(data['pr_epoch'] - 1)
    last_val_tag = 'validation_' + str(data['val_epoch'] - 1)

    if last_val_tag not in data or last_fit_tag not in data:
        last_fit_tag = 'pr_data_' + str(data['train_data']['epoch_best_fit_index'])
        last_val_tag = 'validation_' + str(data['train_data']['epoch_best_fit_index'])

    return last_fit_tag, last_val_tag


"""
    Old data format, is kept for retrocompatibility and will be called if no new data is found on
    YAML dict. Is to be deprecated but I want compatibility with previous test  executed.
"""
def compute_plot_data(data, dataset):
    data_filtered = {}
    try:
        last_fit_tag, last_val_tag = getBestTag(data)
        
        mr_plot = []
        for r_plot in data[last_fit_tag]['r_plot']:
            mr_plot.append((1-np.array(r_plot)).tolist())
        data[last_fit_tag]['mr_plot'] = mr_plot

        # print(f"[{dataset['key']}] recall: {data[last_fit_tag]['r']}")
        # print(f"[{dataset['key']}] Previous mr: {data[last_fit_tag]['mr']}")
        data[last_fit_tag]['mr'] = [(1-data[last_fit_tag]['r'][0])]
        # print(f"[{dataset['key']}] Computed new mr: {data[last_fit_tag]['mr']}")

        ## TBD data will be updated in the metrics function
        if not 'fppi' in data[last_fit_tag]:
            max_f1_index = data[last_fit_tag]['max_f1_index']
            fppi_data = data[last_fit_tag]['fppi_plot'][0]
            fppi_f1max = fppi_data[max_f1_index]
            data[last_fit_tag]['fppi'] = [fppi_f1max]

        if True: # Recompute LAMR for all cases
        # if not 'lamr' in data[last_fit_tag] \
        #     or isinstance(data[last_fit_tag]['lamr'], float) and math.isnan(data[last_fit_tag]['lamr']) \
        #     or isinstance(data[last_fit_tag]['lamr'], float) and data[last_fit_tag]['lamr'] < 0.001 \
        #     or isinstance(data[last_fit_tag]['lamr'], list) and data[last_fit_tag]['lamr'][0] < 0.001:
            lamr = compute_lamr(np.array(data[last_fit_tag]['mr_plot']).flatten(),
                                np.array(data[last_fit_tag]['fppi_plot']).flatten())
            lamr = lamr.tolist()
            data[last_fit_tag]['lamr'] = [lamr]



        validation_data_filtered = {k: v for k, v in data[last_val_tag].items() if k not in ['confusion_matrix']}
        pr_data_filtered = {k: v for k, v in data[last_fit_tag].items() if k not in ['data_store']}
        data_filtered = {'validation_best': validation_data_filtered, 
                         'pr_data_best': pr_data_filtered,
                        'train_data': data['train_data'],'n_images': data['n_images'], 'pretrained': data['pretrained'],
                        'n_classes': data['dataset_info']['nc'], 'dataset_tag': data['dataset_tag'],
                        'device_type': data['system_data']['device_type']
                        }
    except KeyError as e:
        log(f"[{inspect.currentframe().f_code.co_name}] Missing key in results data dict({dataset['key']}): {e}", bcolors.ERROR)
        return {}
    
    return data_filtered

