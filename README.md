# yolo_test_utils

Collection of helper scripts and configuration files to run different tests with YOLO along with dataset manipulation.

## Usage

### Docker 
<details><summary>Click to expand (experimental work)</summary>

#### Requirements
Make sure docker is installed:
``` sh
# Instructions from https://docs.docker.com/engine/install/ubuntu/
sudo apt-get update && sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

To be able to use the NVIDIA GPUs an extra package is needed to interface with Docker:
``` sh
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |\
    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
# Restart the Docker daemon to complete the installation after setting the default runtime:
sudo systemctl stop docker
sudo systemctl start docker
```

GPU access from Docker can be checked with the following command:
``` sh
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```
#### Download Docker image

#### Run Docker image

``` sh
# Need to share paths as volumes in Docker container
export DATASET_ORIGINAL_PATH=${HOME}/eeha/kaist-cvpr15
export DATASET_ANNOTATED_PATH=${HOME}/eeha/kaist-yolo-annotated
export RUN_TEST_PATH=${HOME}/eeha/yolo_test_utils/runs
docker run -it \
    --volume="$DATASET_ORIGINAL_PATH:/root/eeha/kaist-cvpr15" \
    --volume="$DATASET_ANNOTATED_PATH:/root/eeha/kaist-yolo-annotated" \
    --volume="$RUN_TEST_PATH:/root/eeha/yolo_test_utils/runs" \
    enheragu/yolo_tests -c 'all'
```

#### Build Docker image

To be able to use docker without `sudo` permission run the following:
``` sh
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker 
```

Build the docker image:
``` sh
docker build -t enheragu/yolo_tests .
```
> Note: Note that each image is cached. Make sure the reposiroty inside the Image is updated with latest content!

#### Run Docker image

</details>

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

Install requirements from both root and YOLO repo:
``` sh
pip install -r requirements
pip install -r ultralitics_yolov8/requirements.txt
```

Install in editable mode YOLO code (in case you need to edit further stuff):
``` sh
cd ultralitics_yolov8
pip install --editable .
```

Check in config_utils.py to setup the paths to your convinience.

### Execute from repo

> Activate venv before executing

It can be executed through the python entry point:
``` sh
./srcrun_yolo_test.py [options]
```
Theres also a bash script that handles log anvenv activation alltogether (options are the same):

``` sh
source scripts/train_val.sh [options]
```


Available options are:

``` sh
usage: run_yolo_test.py [-h] [-c [CONDITION [CONDITION ...]]] [-o [OPTION [OPTION ...]]] [-m [MODEL [MODEL ...]]] [-d DEVICE] [-ca CACHE] [-p PRETRAINED]
                        [-rm [MODE [MODE ...]]]

Handle operations with YOLOv8, both Validation and Training. Tests will be executed iteratively from all combinations of the configurations provided (condition, option and
model).

optional arguments:
  -h, --help            show this help message and exit
  -c [CONDITION [CONDITION ...]], --condition [CONDITION [CONDITION ...]]
                        Condition from which datasets to use while training. Available options are ['all', 'day', 'night']. Usage: -c item1 item2, -c item3
  -o [OPTION [OPTION ...]], --option [OPTION [OPTION ...]]
                        Option of the dataset to be used. Available options are ['visible', 'lwir', 'hsvt', 'rgbt', 'vths', 'vt', '4ch']. Usage: -c item1 item2, -c item3
  -m [MODEL [MODEL ...]], --model [MODEL [MODEL ...]]
                        Model to be used. Available options are ['yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']. Usage: -c item1 item2, -c item3
  -d DEVICE, --device DEVICE
                        Device to run on, i.e. cuda --device '0' or --device '0,1,2,3' or --device 'cpu'.
  -ca CACHE, --cache CACHE
                        True/ram, disk or False. Use cache for data loading. To load '.npy' files disk option is needed.
  -p PRETRAINED, --pretrained PRETRAINED
                        Whether to use a pretrained model.
  -rm [MODE [MODE ...]], --run-mode [MODE [MODE ...]]
                        Run as validation or test mode. Available options are ['val', 'train']. Usage: -c item1 item2, -c item3

```


## Content
### Python scripts:

- [ ] src/run_yolo_test.py -> Entry point for test execution
- [ ] src/update_datset.py -> Checks whether Kaist dataset is downloaded and formated in the YOLO way. Also checks for a given options to generate different mix of channels (see rgb_thermal_mix.py)
- [ ] src/config_utils.py -> Stores common paths and data to be imported form the rest of the scripts. Also has different methods and utils to be used across different scripts.
- [ ] src/false_positives_check.py
- [ ] src/gather_results.py -> Gathers results from simple_test into a table an composed precission-recall.
- [ ] src/Dataset/kaist_image_label.py -> Visually recreates kaist dataset drawing rectangle and class over each annotation.
- [ ] src/Dataset/kaist_to_yolo_annotations.py -> Creates a new folder with the Kaist dataset ordered and labelled so to be used with YOLO.
- [ ] src/Dataset/rgb_thermal_mix.py -> Creates new dataset images mixing different channels in differents ways.
- [ ] src/YoloExecution/simple_test.py
- [ ] src/YoloExecution/train_yolo.py -> Runs YOLO training (and validation if test datasets are provided) based on a set of models an set of datasets.
- [ ] src/YoloExecution/validation_yolo.py -> Runs YOLO validation test based on a set of models an set of datasets.

### Shell scripts:
- [ ] scripts/train_val.sh -> Script to run validation/train with detached logging to file. Launch from tmux session.

### Configuration files:
- [ ] yolo_config/dataset_condition_option.j2 -> yaml configuration template file with collection of sets from Kaist configured for yolo to train and validate based on coco configuration. To be compiled with a given condition (all, day, night) and option (lwir, visible, ...).
- [ ] yolo_config/yolo_dataset.yaml -> Generic configuration files, as previous.
- [ ] yolo_config/yoloCh4.yaml -> YOLO architecture configuration with four channels to be used.
- [ ] yolo_config/yolo_params.yaml -> YOLO hiperparams.