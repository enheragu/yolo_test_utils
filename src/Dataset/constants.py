

from .image_compression import combine_hsvt, combine_rgbt, combine_4ch, combine_vths, combine_vt, combine_lwir_1ch
from .pca_fa_compression import combine_rgbt_pca_to1ch, combine_rgbt_pca_to2ch, combine_rgbt_pca_to3ch, combine_rgbt_fa_to3ch, combine_rgbt_fa_to2ch, combine_rgbt_fa_to1ch
from pathlib import Path

home = Path.home()

# Path setup for dataset parsing and generation
kaist_path = f"{home}/eeha/kaist-cvpr15"
kaist_sets_path = f"{kaist_path}/imageSets"
kaist_annotation_path = f"{kaist_path}/annotations-xml-new"
kaist_images_path = f"{kaist_path}/images"
kaist_yolo_dataset_path = f"{home}/eeha/kaist-yolo-annotated/" # Output dataset in YOLO format


## Whitelist/blacklist to add or exclude datasets from generation
dataset_blacklist = []
dataset_whitelist = ['train-day-04', 'train-day-20', 'test-day-01', 'test-day-20', 'train-night-02', 'train-night-04', 'test-night-01', 'test-night-20']


# Dict with tag-function to be used when generating different options
dataset_options = {
                    'hsvt': {'merge': combine_hsvt, 'extension': '.png' },
                    'rgbt': {'merge': combine_rgbt, 'extension': '.png' },
                    '4ch': {'merge': combine_4ch, 'extension': '.npy' },
                    'vths' : {'merge': combine_vths, 'extension': '.png' },
                    'vt' : {'merge': combine_vt, 'extension': '.png' },
                    'lwir_1ch' : {'merge': combine_lwir_1ch, 'extension': '.npy' }
                }

fa_pca_options = {'pca_rgbt_1ch' : {'merge': combine_rgbt_pca_to1ch, 'extension': '.npy' },
           'pca_rgbt_2ch' : {'merge': combine_rgbt_pca_to2ch, 'extension': '.npy' },
           'pca_rgbt_3ch' : {'merge': combine_rgbt_pca_to3ch, 'extension': '.png' },
        #    'pca_hsvt_3ch' : {'merge': combine_hsvt_pca_to3ch, 'extension': '.png' }, # -> Result is really bad, makes no sense to look for covariance in that format
           'fa_rgbt_3ch' : {'merge': combine_rgbt_fa_to3ch, 'extension': '.png' },
           'fa_rgbt_2ch' : {'merge': combine_rgbt_fa_to2ch, 'extension': '.npy' },
           'fa_rgbt_1ch' : {'merge': combine_rgbt_fa_to1ch, 'extension': '.npy' }
          }


dataset_options.update(fa_pca_options)
dataset_options_keys = ['visible', 'lwir'] + list(dataset_options.keys())

# Dataset class to take into account when generating YOLO style dataset
class_data = {'kaist_coco': {  'person': 0,  'cyclist': 80, 'people': 81 }, # people does not exist in coco dataset, use 80 as tag
              'kaist': {  'person': 0,  'cyclist': 1, 'people': 2 }
             }
dataset_keys = list(class_data.keys())