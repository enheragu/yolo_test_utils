# yolo_test_utils

Collection of helper scripts and configuration files to run different tests with YOLO along with dataset manipulation.

## Python scripts:
- [ ] src/update_dataset.py -> Checks whether Kaist dataset is downloaded and formated in the YOLO way. Also checks for a given options to generate different mix of channels (see rgb_thermal_mix.py)
- [ ] src/validation_yolo.py -> Runs YOLO validation test based on a set of models an set of datasets.
- [ ] src/train_yolo.py -> Runs YOLO training (and validation if test datasets are provided) based on a set of models an set of datasets.
- [ ] src/kaist_to_yolo_annotations.py -> Creates a new folder with the Kaist dataset ordered and labelled so to be used with YOLO.
- [ ] src/rgb_thermal_mix.py -> Creates new dataset images mixing different channels in differents ways.
- [ ] src/kaist_image_label.py -> Visually recreates kaist dataset drawing rectangle and class over each annotation.
- [ ] src/gather_results.py -> Gathers results from simple_test into a table an composed precission-recall graphs.
- [ ] src/config_utils.py -> Stores common paths and data to be imported form the rest of the scripts. Also has different methods and utils to be used across different scripts.

## Shell scripts:
- [ ] scripts/train_val.sh -> Script to run validation/train with detached logging to file. Launch from tmux session.

## Configuration files:
- [ ] yolo_config/dataset_condition_option.j2 -> yaml configuration template file with collection of sets from Kaist configured for yolo to train and validate based on coco configuration. To be compiled with a given condition (all, day, night) and option (lwir, visible, ...).
- [ ] yolo_config/yolo_dataset.yaml -> Generic configuration files, as previous.
- [ ] yolo_config/yoloCh4.yaml -> YOLO architecture configuration with four channels to be used.
- [ ] yolo_config/yolo_params.yaml -> YOLO hiperparams.