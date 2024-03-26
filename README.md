# yolo_test_utils

This is a collection of helper python/bash scripts to run YOLOv8 in validation/detection mode. This set of scripts help run tests in batch along with a given dataset that is adapted and re-generated if is missing.

Currently it makes use of [YOLOv8](https://github.com/ultralytics/ultralytics) and [Kaist dataset](https://soonminhwang.github.io/rgbt-ped-detection/) with the intention of evaluating the performance of different approaches of image fussion (thermal and RGB data).

## Usage

### Download raw repository

Clone the repository with submodules! 
```sh
    git clone --recurse-submodule
```

> Note: Commands asume your have a terminal oppened and you are located in the root of the repository.

It is recommended if yo create a virtual environment. Paralel to the repository root could be a good idea:

```sh
    python3 -m venv venv
```

Install requirements from both root and YOLO repo. GUI requirements are optional as are only used to run GUI interface:
``` sh
    pip install -r requirements
    pip install -r ultralitics_yolov8/requirements.txt
    pip install -r gui-requirements
```

If GUI is to be used note that some drivers migth collabpse between what OpenCV and PyQt6 use. Please install them manually in the system:
``` sh
    sudo apt install libxcb-cursor0
```

Optionally install in editable mode YOLO code (in case you need to edit further stuff):
``` sh
    cd ultralitics_yolov8
    pip install --editable .
```

Check in Dataset/constants.py to setup the paths to your convinience.

### Execute from repo

The best way to run different tests is checking `scripts/scheduler_options.sh` file. This script includes a set of functions that run tests, load new tests and run GUI. Also to stop and handle logging options. These functions already activate the venv so nothing else is needed.

In case different tests are run in different machines, a device tag can be added to later diferentiate them. Add the following env var to the environment:
``` sh
    export EEHA_TRAIN_DEVICE_TAG="4090"
```

If tests can only run in a given time (noisy server...), add the following env var configuring enabling time. Note that test will start on that time slot, but can end out of it (long tests).
``` sh
    export EEHA_ACTIVE_TEST_TIMETABLE='15:30-06:00'
```

To run the tests the following command is used (theres two variations to explicitly configure GPU devices 0 and 1):
``` sh
    eeha_run_scheduler     # Default device unless configured in option
    eeha_device0_scheduler # Run with GPU device 0 
    eeha_device1_scheduler # Run with GPU device 1
```

Extra test cases can be added with the following command:
``` sh
    eeha_schedule_new_test #option1 ...
```

All four last commands accept the following options as input. It is the same running scheduler with the options or loading the test with schedule_new_test and then running the scheduler apart. schedule_new_test command can load new tests while running.

Available options are:

``` sh
usage: run_scheduled_test.py [-h] [-c [CONDITION ...]] [-o [OPTION ...]] [-m [MODEL ...]] [-ca {ram,disk}] [-p PRETRAINED] [-rm [MODE ...]] [-path PATH_NAME] [-df {kaist_coco,kaist_small,kaist_full}]
                             [-it ITERATIONS] [-b BATCH] [-det [DETERMINISTIC]]

Handle operations with YOLOv8, both Validation and Training. Tests will be executed iteratively from all combinations of the configurations provided (condition, option and model).

options:
  -h, --help            show this help message and exit
  -c [CONDITION ...], --condition [CONDITION ...]
                        Condition from which datasets to use while training. Available options are ['day', 'night', 'all']. Usage: -c item1 item2, -c item3
  -o [OPTION ...], --option [OPTION ...]
                        Option of the dataset to be used. Available options are ['visible', 'lwir', 'hsvt', 'rgbt', '4ch', 'vths', 'vt', 'lwir_npy', 'vt_2ch', 'pca_rgbt_npy', 'fa_rgbt_npy',
                        'pca_full_npy', 'fa_full_npy']. Usage: -c item1 item2, -c item3
  -m [MODEL ...], --model [MODEL ...]
                        Model to be used. Available options are ['yoloCh1x.yaml', 'yoloCh2x.yaml', 'yoloCh3x.yaml', 'yoloCh4x.yaml', 'yolov8x.pt']. Usage: -c item1 item2, -c item3
  -ca {ram,disk}, --cache {ram,disk}
                        True/ram, disk or False. Use cache for data loading. To load '.npy' or '.npz' files disk option is needed.
  -p PRETRAINED, --pretrained PRETRAINED
                        Whether to use a pretrained model.
  -rm [MODE ...], --run-mode [MODE ...]
                        Run as validation or test mode. Available options are ['val', 'train']. Usage: -c item1 item2, -c item3
  -path PATH_NAME, --path-name PATH_NAME
                        Path in which the results will be stored. If set to None a default path will be generated.
  -df {kaist_coco,kaist_small,kaist_full}, --dataset-format {kaist_coco,kaist_small,kaist_full}
                        Format of the dataset to be generated. One of the following: ['kaist_coco', 'kaist_small', 'kaist_full']
  -it ITERATIONS, --iterations ITERATIONS
                        How many repetitions of this test will be performed secuencially.
  -b BATCH, --batch BATCH
                        Batch size when training.
  -det [DETERMINISTIC], --deterministic [DETERMINISTIC]
                        Whether training process makes use of deterministic algorithms or not.

```


## Content

The repository contains different packages and folders:

``` sh
.
├── deprecated
├── media
├── scripts
├── src
│   ├── 0_deprecated
│   ├── Dataset
│   ├── Dataset_review
│   ├── GUI
│   │   └── Widgets
│   ├── utils
│   └── YoloExecution
└── yolo_config
```

- [ ] `deprecated` fodler contains a set of old scripts and stuf that I want to keep over here in case I need them but are not used anymore.
- [ ] `media` folder to store images and logos shown in this file and for documenting purposes.
- [ ] `scripts` folder includes bash scripts. Currently it stores `cheduler_options.sh`
- [ ] `scr` contains all the Python code to run the tests, GUI and all stuff.
- [ ] `scr/0_deprecated` contains deprecated stuf that I want to keep close :)
- [ ] `scr/Dataset` contains scripts to handle the Dataset. Kaist dataset needs to be translated to YOLO format and some fusion images are to be generated.
- [ ] `scr/Dataset_review` contains a script to review if the dataset generation went according to what was expected (number of images, instances and so on).
- [ ] `scr/GUI` contains the code of the results analyzer tool.
- [ ] `scr/GUI/Widgets` contains extended and custom PyQt widgets that are used in the GUI
- [ ] `scr/utils` contains extra utilities such as loggin, tagging and YAML handling that are used in the code.
- [ ] `scr/YoloExecution` contains both train and validation connecting point to YOLO repository.
- [ ] `yolo_config` includes configuration files both for dataset, model and hyperparameters. Note that some CFG are templated to be generated during execution.


### Python scripts:

Scripts that can be executed are (see first section about how to run the utilities) :
- [ ] `src/run_scheduled_test.py` run testing system.
- [ ] `src/gui_plot_results.py` runs GUI.
- [ ] `src/test_scheduler.py` script to schedule new tests.
- [ ] `src/compresslabel_folder.py` compres label folder outputed by YOLO (if many tests are performed it can take quite some disk space). This functionality is integrated into the toolchain so is not something that needs to be run independly.
- [ ] `scr/Dataset_review/review_datset.py` extracts some information about number of images (if theyr are repeated, how many) and instances of given classes.
- [ ] `src/Dataset/update_dataset.py` checks if dataset is available for options requested. If not dataset is downloaded, adapted to YOLO expected format and options requested are generated.
- [ ] `src/Dataset/kaist_to_yolo_annotations.py` adapts downloaded dataset to YOLO format.
- [ ] `src/Dataset/rgb_thermal_mix.py`: taking that dataset already exists in YOLO format, perform the different fusion methods requested.

> Please check input options for each individual case.