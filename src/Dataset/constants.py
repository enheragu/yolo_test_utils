

from .image_compression import combine_hsvt, combine_rgbt, combine_4ch, combine_vths, combine_vt, combine_lwir_npy, combine_vt_2ch
from .pca_fa_compression import combine_rgbt_pca_to3ch, combine_rgbt_fa_to3ch, combine_rgbt_pca_full, combine_rgbt_fa_full, preprocess_rgbt_pca_full, preprocess_rgbt_fa_full
from pathlib import Path

home = Path.home()
repo_path = f"{home}/eeha/yolo_test_utils"

# Path setup for dataset parsing and generation
kaist_path = f"{home}/eeha/kaist-cvpr15"
kaist_sets_paths = [f"{kaist_path}/imageSets", f"{repo_path}/kaist_imageSets"]
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
                    'vt_2ch' : {'merge': combine_vt_2ch, 'extension': '.npz' }
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
class_data = {'kaist_coco': {  'person': 0,  'cyclist': 80, 'people': 81 }, # people does not exist in coco dataset, use 80 as tag
              'kaist_small': {  'person': 0,  'cyclist': 1, 'people': 2 },
              'kaist_full': {  'person': 0,  'cyclist': 1, 'people': 2 },
             }

## Templates for TMP YOLO dataset configuration
# kaist_coco -> makes use of kaist_small but with class dict as defined by coco
# kaist_small -> kaist with reduced version (less images)
# kaist_full -> kaist with all images
templates_cfg = {'kaist_coco': f"{dataset_config_path}/dataset_kaist_coco_option.j2",
                 'kaist_small': f"{dataset_config_path}/dataset_kaist_small_option.j2",
                 'kaist_full': f"{dataset_config_path}/dataset_kaist_full_option.j2"
                 }

dataset_keys = list(class_data.keys())


condition_list_default = ['day','night','all']
option_list_default = dataset_options_keys
model_list_default = ['yoloCh1x.yaml','yoloCh2x.yaml','yoloCh3x.yaml','yoloCh4x.yaml','yolov8x.pt'] #['yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
dataset_tags_default = dataset_keys   # Just list of availables :)