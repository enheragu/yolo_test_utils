#!/usr/bin/env python3
# encoding: utf-8

import os 
from pathlib import Path
import glob
from datetime import datetime

from yolo.utils import DEFAULT_CFG
import yolo.v8.detect as yolo_detc
from yolo.cfg import get_cfg
from ultralytics import YOLO


from utils import parseYaml, dumpYaml, log, bcolors
from Dataset.constants import get_fusion_method_metadata


def _opts_to_plain_dict(opts):
    """Serialize argparse.Namespace-like options without Python object YAML tags."""
    if isinstance(opts, dict):
        return dict(opts)
    if hasattr(opts, '__dict__'):
        return dict(vars(opts))
    return opts


def TestValidateYolo(dataset, yolo_model, path_name, opts, log_file_path, option):
    start_time = datetime.now()

    log("-------------------------------------")
    log(f"[{yolo_model}] - Check {dataset} dataset")
    data = parseYaml(dataset)
    log(f"[{yolo_model}] - Validation datasets: {data['val']}")      
    
    images_png = 0
    images_npy = 0
    yaml_data = {'n_images': {}}
    for val_dataset in data['val']:
        data_path = f"{data['path']}/{val_dataset}/images"
        images_png += len(glob.glob1(data_path,"*.png"))
        images_npy += len(glob.glob1(data_path,"*.npy"))
        images_npy += len(glob.glob1(data_path,"*.npz"))
    if images_png:
        log(f"[{yolo_model}] - Validation with {images_png} png images")
    if images_npy:
        log(f"[{yolo_model}] - Validation with {images_npy} npy images")            

    yaml_data['n_images']['val'] = images_png + images_npy
    yaml_data['dataset_tag'] = opts.dformat
    yaml_data['thermal_equalization'] = opts.thermal_eq
    yaml_data['rgb_equalization'] = opts.rgb_eq

    fusion_meta = get_fusion_method_metadata(option)
    yaml_data['fusion_metadata'] = {}
    yaml_data['fusion_metadata']['method'] = fusion_meta['method']
    yaml_data['fusion_metadata']['method_version'] = fusion_meta['version']
    yaml_data['fusion_metadata']['method_symbol'] = fusion_meta['merge_symbol']
    yaml_data['fusion_metadata']['method_module'] = fusion_meta['merge_module']

    args = {} 
    # args['project'] = 'detection'
    args['mode'] = 'val'
    args['name'] = path_name
    args['model'] = yolo_model
    args['data'] = dataset
    # args['imgsz'] = 640
    args['save'] = True
    args['save_txt'] = True
    args['verbose'] = True
    args['save_conf'] = True
    args['save_json'] = True
    args['device'] = opts.device
    # args['save_hybrid'] = True -> PROBLEMS WITH TENSOR SIZE

    # Lean output for re-validation / cross-eval (when --result-tag is set): the saved label .txt are
    # redundant with predictions.json (reconstruct_results.py reads the json), so skip them. Plots stay
    # ON because the val_batch mosaics are NOT regenerable once the dataset npz are deleted; the curve
    # PNGs (which ARE reconstructible from pr_data in results.yaml) are pruned later by a post-backup cleanup.
    if getattr(opts, 'result_tag', None):
        args['save_txt'] = False

    yaml_data['output_log_file'] = log_file_path

    args = get_cfg(cfg=DEFAULT_CFG, overrides=args)
    # opt = yolo_detc.parse_opt()    
    # log(DEFAULT_CFG)
    # yolo_detc.val()
    validator = yolo_detc.DetectionValidator(args=args)
    validator(model=args.model)

    results_path = Path(validator.save_dir) / f'results.yaml'
    existing = parseYaml(results_path) or {}
    existing.update(yaml_data)
    dumpYaml(results_path, existing)
    run_opts_data = _opts_to_plain_dict(opts)
    dumpYaml(Path(validator.save_dir) / f'run_opts.yaml', run_opts_data, 'w')

    log(f"[{yolo_model}] - Dataset processing took {datetime.now() - start_time} (h/min/s)")
    log("-------------------------------------")
           

if __name__ == '__main__':
    pass
    # condition_list, option_list, model_list, device, cache, pretrained, _ = handleArguments()
    # TestValidateYolo(condition_list, option_list, model_list, device, cache, pretrained)