#!/usr/bin/env bash

## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )


test_finished() {
    curl \
    -d "Finished execution of test cases: $1. Finished with status $2" \
    -H "Title: Execution finished" \
    -H "Priority: default" \
    -H "Tags: +1" \
    ntfy.sh/eeha_training_test_battery
}


# source $SCRIPT_PATH/train_val.sh -c 'day' 'night' -o 'hsvt' 'vths' 'vt' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'hsvt vths vt' $?
# source $SCRIPT_PATH/train_val.sh -c 'night' -o 'vt' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'night_vt' $?
# source $SCRIPT_PATH/train_val.sh -c 'day' 'night' -o 'visible' 'lwir' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'visible lwir' $?
# source $SCRIPT_PATH/train_val.sh -c 'day' 'night' -o 'rgbt' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'rgbt' $?
# source $SCRIPT_PATH/train_val.sh -c 'night' 'day'  -o '4ch' -m 'yoloCh4x.yaml' -rm 'train' --pretrained False --cache disk
# test_finished '4ch' $?

# source $SCRIPT_PATH/train_val.sh -c 'night' 'day'  -o 'pca_rgbt_1ch' 'fa_rgbt_1ch' -m 'yoloCh1x.yaml' -rm 'train' --pretrained False --cache disk
# test_finished 'pca_rgbt_1ch fa_rgbt_1ch' $?
# source $SCRIPT_PATH/train_val.sh -c 'night' 'day'  -o 'pca_rgbt_2ch' 'fa_rgbt_2ch' -m 'yoloCh2x.yaml' -rm 'train' --pretrained False --cache disk
# test_finished 'pca_rgbt_2ch fa_rgbt_2ch' $?
# source $SCRIPT_PATH/train_val.sh -c 'day' 'night' -o 'pca_rgbt_3ch' 'fa_rgbt_3ch' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'pca_rgbt_3ch fa_rgbt_3ch' $?


# source $SCRIPT_PATH/train_val.sh -c 'night' -o 'lwir' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'nigth lwir' $?
# source $SCRIPT_PATH/train_val.sh -c 'night' 'day'  -o 'lwir_1ch' -m 'yoloCh1x.yaml' -rm 'train' --pretrained False --cache disk
# test_finished 'lwir_1ch' $?
# source $SCRIPT_PATH/train_val.sh -c 'night' -o 'pca_rgbt_3ch' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'night pca_rgbt_3ch' $?
source $SCRIPT_PATH/train_val.sh -c 'day' 'night' -o 'fa_rgbt_3ch' -m 'yolov8x.pt' -rm 'train' --pretrained True
test_finished 'fa_rgbt_3ch' $?



test_finished "all script finished" ":)"