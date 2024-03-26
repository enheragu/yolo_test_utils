#!/usr/bin/env bash

## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
    SOURCE=$(readlink "$SOURCE")
    [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

export NTFY_TOPIC="eeha_training_test_battery"


# source $SCRIPT_PATH/train_val.sh -c 'night' 'day' -o 'hsvt' 'vths' 'vt' 'visible' 'lwir' 'rgbt' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'hsvt vths vt visible lwir rgbt' $?

# source $SCRIPT_PATH/train_val.sh -c 'night' 'day'  -o 'lwir_1ch' -m 'yoloCh1x.yaml' -rm 'train' --pretrained False --cache disk
# test_finished 'lwir_1ch' $?
# source $SCRIPT_PATH/train_val.sh -c 'night' 'day'  -o 'pca_rgbt_1ch' 'fa_rgbt_1ch' -m 'yoloCh1x.yaml' -rm 'train' --pretrained False --cache disk
# test_finished 'pca_rgbt_1ch fa_rgbt_1ch' $?

# source $SCRIPT_PATH/train_val.sh -c 'night' 'day'  -o 'pca_rgbt_2ch' 'fa_rgbt_2ch' -m 'yoloCh2x.yaml' -rm 'train' --pretrained False --cache disk
# test_finished 'pca_rgbt_2ch fa_rgbt_2ch' $?

# source $SCRIPT_PATH/train_val.sh -c 'night' 'day' -o 'pca_rgbt_3ch' 'fa_rgbt_3ch' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'pca_rgbt_3ch fa_rgbt_3ch' $?

# source $SCRIPT_PATH/train_val.sh -c 'night' 'day'  -o '4ch' -m 'yoloCh4x.yaml' -rm 'train' --pretrained False --cache disk
# test_finished '4ch' $?

#################################################
# Test if theres differences when training from #
#  scratch than training from pretrained model  #
#################################################
# source $SCRIPT_PATH/train_val.sh -c 'night' 'day' -o 'visible' 'lwir' -m 'yoloCh3x.yaml' -rm 'train' --pretrained False
# test_finished 'visible lwir without pretraining with day and night condition datasets' $?

# source $SCRIPT_PATH/train_val.sh -c 'all' -o 'visible' 'rgbt' 'vths' 'vt' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'vths vt visible rgbt (good day candidates), with all conditions datasets' $?

# Also good VTHS, RGBT and VT that are already being used
# source $SCRIPT_PATH/train_val.sh -c 'all' -o 'lwir' 'pca_rgbt_3ch' -m 'yolov8x.pt' -rm 'train' --pretrained True
# test_finished 'lwir pca_rgbt_3ch (good night candidates), with all conditions datasets' $?




## Variance between different training results
for i in {1..2}
do
    source $SCRIPT_PATH/train_val.sh -c 'day' -o 'visible' -m 'yoloCh3x.yaml' -rm 'train' --pretrained False \
                                   --path-name "variance_day_visible_kaist_train" --dataset-format 'kaist'
done

# for i in {1..10}
# do
#     source $SCRIPT_PATH/train_val.sh -c 'night' -o 'lwir' -m 'yoloCh3x.yaml' -rm 'train' --pretrained False \
#                                    --path-name "variance_night_lwir_kaist_train" --dataset-format 'kaist'
# done

# for i in {1..10}
# do
#     source $SCRIPT_PATH/train_val.sh -c 'day' -o 'vt' -m 'yoloCh3x.yaml' -rm 'train' --pretrained False \
#                                    --path-name "variance_day_vt_kaist_train" --dataset-format 'kaist'
# done



battery_test_finished() {
    curl \
    -d "Finished execution of all test cases." \
    -H "Title: ¡Execution finished!" \
    -H "Priority: default" \
    -H "Tags: +1,partying_face" \
    ntfy.sh/eeha_training_test_battery
}

battery_test_finished