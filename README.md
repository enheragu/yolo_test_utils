# yolo_test_utils

Collection of helper scripts and configuration files to run different tests with YOLO along with dataset manipulation.

## Python scripts:
- [ ] src/validation_yolo.py -> Runs YOLO validation test based on a set of models an set of datasets.
- [ ] src/train_yolo.py -> Runs YOLO training (and validation if test datasets are provided) based on a set of models an set of datasets.
- [ ] src/kaist_to_yolo_annotations.py -> Creates a new folder with the Kaist dataset ordered and labelled so to be used with YOLO.
- [ ] src/kaist_image_label.py -> Visually recreates kaist dataset drawing rectangle and class over each annotation.
- [ ] src/gather_results.py -> Gathers results from simple_test into a table an composed precission-recall graphs.
- [ ] src/config.py -> Stores common paths and data to be imported form the rest of the scripts.

## Shell scripts:
- [ ] scripts/train_val.sh -> Script to run validation/train with detached logging to file. Launch from tmux session.

## Configuration files:
- [ ] yolo_config/dataset_day_lwir.yaml -> yaml configuration file with collection of sets from Kaist configured for yolo to train and validate based on coco configuration. ALl sets combining lwir images from day scenarios.
- [ ] yolo_config/dataset_day_visible.yaml -> (same day visible).
- [ ] yolo_config/dataset_night_lwir.yaml -> (same nigth lwir).
- [ ] yolo_config/dataset_night_visible.yaml -> (same night visible).
- [ ] yolo_config/yolo_dataset.yaml -> Generic configuration files, as previous.
- [ ] yolo_config/yolo_eeha.yaml -> YOLO architecture configuration.
- [ ] yolo_config/yolo_params.yaml -> YOLO hiperparams.