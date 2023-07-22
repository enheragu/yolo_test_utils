#!/usr/bin/env bash

## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

source $SCRIPT_PATH/../../venv/bin/activate


now=`date +"%Y_%m_%d-%H_%M_%S"`

LOG_PATH=$SCRIPT_PATH/../runs/
mkdir -p $LOG_PATH
echo "Logging execution to $LOG_PATH/${now}_test_yolo.log" 
(cd $SCRIPT_PATH/.. && time ./src/run_yolo_test.py $@ 2>&1 | tee -a $LOG_PATH/${now}_test_yolo.log)
echo "Logged execution to $LOG_PATH/${now}_test_yolo.log. All YOLO output can be found in $LOG_PATH/detect/"


# -c 'day' 'night' -o 'visible' 'lwir' 'rgbt' -m 'yolov8x.pt' -rm 'train' 'val' --pretrained True
# -c 'day' 'night' -o 'hsvt' 'vths' 'vt' -m 'yolov8x.pt' -rm 'train' 'val' --pretrained True


    
# (cd $SCRIPT_PATH/.. && time \
# ./src/run_yolo_test.py -c 'night' \
#                     -o '4ch' \
#                     -m 'yoloCh4x.yaml' \
#                     --device 'cpu' \
#                     --cache 'disk' \
#                     --pretrained False \
#                     --rm 'train' \
#  2>&1)
