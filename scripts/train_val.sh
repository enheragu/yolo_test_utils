#!/usr/bin/env bash
echo "\n\n\n---------------------------------------------------------------------------"
## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

source $SCRIPT_PATH/../../venv/bin/activate


NOW=`date +"%Y_%m_%d-%H_%M_%S"`
LOG_PATH=$SCRIPT_PATH/../runs/exec_log
LOG_FILENAME="${LOG_PATH}/${NOW}_test_yolo.log"

# if file already exists ${LOG_PATH}/../now_executing${REPEATED}.log
if [ ! -L "${LOG_PATH}/../now_executing.log" ]; then 
REPEATED=''
else
echo "Another process is already running, shortcut to log to a different name"
REPEATED='_1'
fi

# Ensure path and file exists and make shortcut. If theres already one test being executed
# add a surname to file shortcut names 
mkdir -p $LOG_PATH
touch ${LOG_FILENAME}
ln -s ${LOG_FILENAME} ${LOG_PATH}/../now_executing${REPEATED}.log

echo "Logging execution to ${LOG_FILENAME}" 
echo "Executing: $0 $@" >> ${LOG_FILENAME}
(cd $SCRIPT_PATH/.. && time ./src/run_yolo_test.py $@ 2>&1 | tee -a ${LOG_FILENAME})
echo "Logged execution to ${LOG_FILENAME}. All YOLO output can be found in $LOG_PATH/"

# Remove previous shortcut and make new one
ln -s ${LOG_FILENAME} ${LOG_PATH}/../latest${REPEATED}.log
rm ${LOG_PATH}/../now_executing${REPEATED}.log


echo "---------------------------------------------------------------------------\n\n\n"
# -c 'day' 'night' -o 'visible' 'lwir' 'rgbt' -m 'yolov8x.pt' -rm 'train' 'val' --pretrained True
# -c 'day' 'night' -o 'hsvt' 'vths' 'vt' -m 'yolov8x.pt' -rm 'train' 'val' --pretrained True


# Executed:
# source scripts/train_val.sh -c 'day' 'night' -o 'hsvt' 'vths' 'vt' -m 'yolov8x.pt' -rm 'val' 'train' --pretrained True
# source scripts/train_val.sh -c 'day' 'night' -o 'visible' 'lwir' -m 'yolov8x.pt' -rm 'train' --pretrained True; \
# source scripts/train_val.sh -c 'day' 'night' -o 'rgbt' -m 'yolov8x.pt' -rm 'val' 'train' --pretrained True