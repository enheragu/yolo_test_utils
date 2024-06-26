

from .image_compression import combine_hsvt, combine_rgbt, combine_4ch, combine_vths, combine_vt, combine_lwir_npy, combine_vt_2ch
from .image_compression import   combine_vths_v2, combine_vths_v3, combine_rgbt_v2
from .pca_fa_compression import combine_rgbt_pca_to3ch, combine_rgbt_fa_to3ch, combine_rgbt_pca_full, combine_rgbt_fa_full, preprocess_rgbt_pca_full, preprocess_rgbt_fa_full
from pathlib import Path

home = Path.home()
repo_path = f"{home}/eeha/yolo_test_utils"

# Path setup for dataset parsing and generation
kaist_path = f"{home}/eeha/kaist-cvpr15"
kaist_sets_paths = [f"{repo_path}/kaist_imageSets", f"{kaist_path}/imageSets"]
kaist_annotation_path = f"{kaist_path}/annotations-xml-new"
kaist_images_path = f"{kaist_path}/images"
kaist_yolo_dataset_path = f"{home}/eeha/kaist-yolo-annotated/" # Output dataset in YOLO format

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
                    '4ch': {'merge': combine_4ch, 'extension': '.npy' },
                    'vths' : {'merge': combine_vths, 'extension': '.png' },
                    'vt' : {'merge': combine_vt, 'extension': '.png' },
                    'lwir_npy' : {'merge': combine_lwir_npy, 'extension': '.npz' },
                    'vt_2ch' : {'merge': combine_vt_2ch, 'extension': '.npz' },

                    'rgbt_v2' : {'merge': combine_rgbt_v2, 'extension': '.png' },
                    'vths_v2' : {'merge': combine_vths_v2, 'extension': '.png' },
                    'vths_v3' : {'merge': combine_vths_v3, 'extension': '.png' }
                  }

fa_pca_options = {'pca_rgbt_npy' : {'merge': combine_rgbt_pca_to3ch, 'extension': '.npz' },
                  'fa_rgbt_npy' : {'merge': combine_rgbt_fa_to3ch, 'extension': '.npz' },
                  'pca_full_npy' : {'merge': combine_rgbt_pca_full, 'extension': '.npz', 'preprocess': preprocess_rgbt_pca_full },
                  'fa_full_npy' : {'merge': combine_rgbt_fa_full, 'extension': '.npz', 'preprocess': preprocess_rgbt_fa_full }
                  }
         # Modified YOLO dataloader so it only loads needed stuff
         #   'pca_rgbt_1ch' : {'merge': combine_rgbt_pca_to1ch, 'extension': '.npy' },
         #   'pca_rgbt_2ch' : {'merge': combine_rgbt_pca_to2ch, 'extension': '.npy' },
         #   'pca_hsvt_3ch' : {'merge': combine_hsvt_pca_to3ch, 'extension': '.png' }, # -> Result is really bad, makes no sense to look for covariance in that format
         #   'fa_rgbt_2ch' : {'merge': combine_rgbt_fa_to2ch, 'extension': '.npy' },
         #   'fa_rgbt_1ch' : {'merge': combine_rgbt_fa_to1ch, 'extension': '.npy' }
          


dataset_options.update(fa_pca_options)
dataset_options_keys = ['visible', 'lwir'] + list(dataset_options.keys())

# Dataset class to take into account when generating YOLO style dataset
default_kaist = {  'person': 0,  'cyclist': 1, 'people': 2 }
class_data = {'kaist_coco': {  'person': 0,  'cyclist': 80, 'people': 81 }, # people does not exist in coco dataset, use 80 as tag
              'kaist_small': default_kaist,
              'kaist_full': default_kaist,
              'kaist_90_10': default_kaist,
              'kaist_80_20': default_kaist,
              'kaist_70_30': default_kaist,
              'kaist_debug': default_kaist
             }

## Templates for TMP YOLO dataset configuration
# kaist_coco -> makes use of kaist_small but with class dict as defined by coco
# kaist_small -> kaist with reduced version (less images)
# kaist_full -> kaist with all images
templates_cfg = {'kaist_coco': {'template': f"{dataset_config_path}/dataset_kaist_coco_option.j2"},
                 'kaist_small': {'template': f"{dataset_config_path}/dataset_kaist_small_option.j2"},
                 'kaist_full': {'template': f"{dataset_config_path}/dataset_kaist_full_option.j2"},
                 'kaist_90_10': {'template': f"{dataset_config_path}/dataset_kaist_percent_option.j2", 'extra': {'percent': '90_10'}}, # Extra arguments that can be provided to the template
                 'kaist_80_20': {'template': f"{dataset_config_path}/dataset_kaist_percent_option.j2", 'extra': {'percent': '80_20'}}, # Extra arguments that can be provided to the template
                 'kaist_70_30': {'template': f"{dataset_config_path}/dataset_kaist_percent_option.j2", 'extra': {'percent': '70_30'}}, # Extra arguments that can be provided to the template
                 'kaist_debug': {'template': f"{dataset_config_path}/dataset_kaist_percent_option.j2", 'extra': {'percent': 'debug'}} # Extra arguments that can be provided to the template. Just for debugging training/val process
                 }

dataset_keys = list(class_data.keys())


condition_list_default = ['day','night','all']
option_list_default = dataset_options_keys
model_list_default = ['yoloCh1m.yaml','yoloCh2m.yaml','yoloCh3m.yaml','yoloCh4m.yaml','yolov8x.pt'] #['yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
dataset_tags_default = dataset_keys   # Just list of availables :)

dataset_generated_cache = f'{kaist_yolo_dataset_path}dataset_options.cache'