

from .fusion_methods.static_image_compression import combine_hsvt, combine_rgbt, combine_4ch, combine_vths, combine_vt, combine_lwir_npy, combine_vt_2ch, combine_rgbtalpha
from .fusion_methods.static_image_compression import combine_4ch_visible, combine_4ch_lwir
from .fusion_methods.static_image_compression import   combine_vths_v2, combine_vths_v3, combine_rgbt_v2
from .fusion_methods.pca_fa_compression import combine_rgbt_pca_to3ch, combine_rgbt_fa_to3ch, combine_rgbt_pca_full, combine_rgbt_fa_full, preprocess_rgbt_pca_full, preprocess_rgbt_fa_full, combine_rgbt_pca_to1ch, combine_rgbt_fa_to1ch, combine_rgbt_alpha_pca_to3ch
from .fusion_methods.wavelets_mdmr_compression import combine_hsvt_wavelet, combine_rgb_wavelet, combine_hsv_curvelet, combine_rgb_curvelet
from .fusion_methods.local_filter_fusion import combine_rgbt_ssim, combine_rgbt_superpixel, combine_rgbt_sobel_weighted
from pathlib import Path

home = Path.home()
repo_path = f"{home}/eeha/yolo_test_utils"

# Path setup for dataset parsing and generation
kaist_path = f"{home}/eeha/kaist-cvpr15"
kaist_sets_paths = [f"{repo_path}/ImageSets/kaist_imageSets", f"{kaist_path}/imageSets"]
kaist_annotation_path = f"{kaist_path}/annotations-xml-new"
# kaist_annotation_path = f"{kaist_path}/annotations-kaist-paired" # KAIST-Paired from https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Weakly_Aligned_Cross-Modal_Learning_for_Multispectral_Pedestrian_Detection_ICCV_2019_paper.pdf
kaist_images_path = f"{kaist_path}/images"
kaist_yolo_dataset_path = f"{home}/eeha/kaist-yolo-annotated/" # Output dataset in YOLO format

llvip_path = f"{home}/eeha/LLVIP"
llvip_annotation_path = f"{llvip_path}/Annotations"
llvip_sets_paths = [f"{repo_path}/ImageSets/llvip_imageSets"]
llvip_images_path = f"{llvip_path}"
llvip_yolo_dataset_path = f"{home}/eeha/llvip-yolo-annotated/" # Output dataset in YOLO format


dataset_config_path = f"{repo_path}/yolo_config"

labels_folder_name = "labels"
images_folder_name = "images"
lwir_folder_name = "lwir"
visible_folder_name = "visible"

## Whitelist/blacklist to add or exclude datasets from generation
dataset_blacklist = ['test-all-01-Seq']
dataset_whitelist = [] #['train-day-04', 'train-day-20', 'test-day-01', 'test-day-20', 'train-night-02', 'train-night-04', 'test-night-01', 'test-night-20']


# Dict with tag-function to be used when generating different options
dataset_options = {
                    'hsvt': {'merge': combine_hsvt, 'extension': '.png' },
                    'rgbt': {'merge': combine_rgbt, 'extension': '.png' },
                    'alphat_rgbt' : {'merge': combine_rgbtalpha, 'extension': '.png' },
                    '4ch': {'merge': combine_4ch, 'extension': '.npz' },
                    'vths' : {'merge': combine_vths, 'extension': '.png' },
                    'vt' : {'merge': combine_vt, 'extension': '.png' },
                    'lwir_npy' : {'merge': combine_lwir_npy, 'extension': '.npz' },
                    'vt_2ch' : {'merge': combine_vt_2ch, 'extension': '.npz' },

                    'rgbt_v2' : {'merge': combine_rgbt_v2, 'extension': '.png' },
                    'vths_v2' : {'merge': combine_vths_v2, 'extension': '.png' },
                    'vths_v3' : {'merge': combine_vths_v3, 'extension': '.png' },

                    '4ch_visible': {'merge': combine_4ch_visible, 'extension': '.npz' },
                    '4ch_lwir': {'merge': combine_4ch_lwir, 'extension': '.npz' }
                  }

fa_pca_options = {'pca' : {'merge': combine_rgbt_pca_to3ch, 'extension': '.npz' },
                  'alpha_pca': {'merge': combine_rgbt_alpha_pca_to3ch, 'extension': '.npz'},
                  'fa' : {'merge': combine_rgbt_fa_to3ch, 'extension': '.npz' },
                  'pca_1ch' : {'merge': combine_rgbt_pca_to1ch, 'extension': '.npz' },
                  'fa_1ch' : {'merge': combine_rgbt_fa_to1ch, 'extension': '.npz' },
                  # Full takes decomposition of all images and then applies transform to each image
                  'pca_full' : {'merge': combine_rgbt_pca_full, 'extension': '.npz', 'preprocess': preprocess_rgbt_pca_full },
                  'fa_full' : {'merge': combine_rgbt_fa_full, 'extension': '.npz', 'preprocess': preprocess_rgbt_fa_full }
                  }

wavelets_options = {
                    # 'wavelet_hsvt' : {'merge': combine_hsvt_wavelet, 'extension': '.npz'},
                    'wavelet' : {'merge': combine_rgb_wavelet, 'extension': '.npz'},
                    # 'curvelet_hsvt' : {'merge': combine_hsv_curvelet, 'extension': '.npz'},
                    'curvelet' : {'merge': combine_rgb_curvelet, 'extension': '.npz'}
                  }

local_filter_options = {
                    'ssim' : {'merge': combine_rgbt_ssim, 'extension': '.png'},
                    'superpixel' : {'merge': combine_rgbt_superpixel, 'extension': '.png'},
                    'sobel_weighted' : {'merge': combine_rgbt_sobel_weighted, 'extension': '.png'}
                  }
         # Modified YOLO dataloader so it only loads needed stuff
         #   'pca_rgbt_1ch' : {'merge': combine_rgbt_pca_to1ch, 'extension': '.npy' },
         #   'pca_rgbt_2ch' : {'merge': combine_rgbt_pca_to2ch, 'extension': '.npy' },
         #   'pca_hsvt_3ch' : {'merge': combine_hsvt_pca_to3ch, 'extension': '.png' }, # -> Result is really bad, makes no sense to look for covariance in that format
         #   'fa_rgbt_2ch' : {'merge': combine_rgbt_fa_to2ch, 'extension': '.npy' },
         #   'fa_rgbt_1ch' : {'merge': combine_rgbt_fa_to1ch, 'extension': '.npy' }
          


dataset_options.update(fa_pca_options)
dataset_options.update(wavelets_options)
dataset_options.update(local_filter_options)
dataset_options_keys = ['visible', 'lwir'] + list(dataset_options.keys())

# Dataset class to take into account when generating YOLO style dataset
default_kaist = {  'person': 0,  'cyclist': 1, 'people': 2 }
class_data = {'coco': {  'person': 0,  'bicycle': 1,  'car': 2,  'motorcycle': 3,  'airplane': 4,  'bus': 5,  'train': 6,  'truck': 7,  'boat': 8,  'traffic light': 9,  'fire hydrant': 10,  'stop sign': 11,  'parking meter': 12,  'bench': 13,  'bird': 14,  'cat': 15,  'dog': 16,  'horse': 17,  'sheep': 18,  'cow': 19,  'elephant': 20,  'bear': 21,  'zebra': 22,  'giraffe': 23,  'backpack': 24,  'umbrella': 25,  'handbag': 26,  'tie': 27,  'suitcase': 28,  'frisbee': 29,  'skis': 30,  'snowboard': 31,  'sports ball': 32,  'kite': 33,  'baseball bat': 34,  'baseball glove': 35,  'skateboard': 36,  'surfboard': 37,  'tennis racket': 38,  'bottle': 39,  'wine glass': 40,  'cup': 41,  'fork': 42,  'knife': 43,  'spoon': 44,  'bowl': 45,  'banana': 46,  'apple': 47,  'sandwich': 48,  'orange': 49,  'broccoli': 50,  'carrot': 51,  'hot dog': 52,  'pizza': 53,  'donut': 54,  'cake': 55,  'chair': 56,  'couch': 57,  'potted plant': 58,  'bed': 59,  'dining table': 60,  'toilet': 61,  'tv': 62,  'laptop': 63,  'mouse': 64,  'remote': 65,  'keyboard': 66,  'cell phone': 67,  'microwave': 68,  'oven': 69,  'toaster': 70,  'sink': 71,  'refrigerator': 72,  'book': 73,  'clock': 74,  'vase': 75,  'scissors': 76,  'teddy bear': 77,  'hair drier': 78,  'toothbrush},': 79},
              'kaist_coco': {  'person': 0,  'cyclist': 80, 'people': 81 }, # people does not exist in coco dataset, use 80 as tag
              'kaist_small': default_kaist,
              'kaist_full': default_kaist,
              'kaist_90_10': default_kaist,
              'kaist_80_20': default_kaist,
              'kaist_70_30': default_kaist,
              'kaist_debug': default_kaist,
              'llvip_80_20': default_kaist,
              'llvip_4ch_latesplit_80_20': default_kaist
             }

## Templates for TMP YOLO dataset configuration
# kaist_coco -> makes use of kaist_small but with class dict as defined by coco
# kaist_small -> kaist with reduced version (less images)
# kaist_full -> kaist with all images
templates_cfg = {'coco': {'template': f"{dataset_config_path}/coco.yaml"},
                 'kaist_coco': {'template': f"{dataset_config_path}/dataset_kaist_coco_option.j2"},
                 'kaist_small': {'template': f"{dataset_config_path}/dataset_kaist_small_option.j2"},
                 'kaist_full': {'template': f"{dataset_config_path}/dataset_kaist_full_option.j2"},
                 'kaist_90_10': {'template': f"{dataset_config_path}/dataset_kaist_percent_option.j2", 'extra': {'percent': '90_10'}}, # Extra arguments that can be provided to the template
                 'kaist_80_20': {'template': f"{dataset_config_path}/dataset_kaist_percent_option.j2", 'extra': {'percent': '80_20'}}, # Extra arguments that can be provided to the template
                 'kaist_70_30': {'template': f"{dataset_config_path}/dataset_kaist_percent_option.j2", 'extra': {'percent': '70_30'}}, # Extra arguments that can be provided to the template
                 'kaist_debug': {'template': f"{dataset_config_path}/dataset_kaist_percent_option.j2", 'extra': {'percent': 'debug'}}, # Extra arguments that can be provided to the template. Just for debugging training/val process
                 'llvip_80_20': {'template': f"{dataset_config_path}/dataset_llvip_percent_option.j2", 'extra': {'percent': '80_20'}},
                 'llvip_4ch_latesplit_80_20': {'template': f"{dataset_config_path}/dataset_llvip_percent_option_4ch_latesplit.j2", 'extra': {'percent': '80_20'}}
                 }

dataset_keys = list(class_data.keys())


condition_list_default = ['day','night','all']
option_list_default = dataset_options_keys
model_list_default = ['yoloNoTrained.pt','yoloCh1m.yaml','yoloCh2m.yaml','yoloCh3m.yaml','yoloCh4m.yaml','yolov8x.pt'] #['yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
dataset_tags_default = dataset_keys   # Just list of availables :)

dataset_generated_cache = f'{repo_path}/.dataset_generation_options.cache'