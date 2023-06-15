#!/usr/bin/env bash

## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

LOG_PATH=$SCRIPT_PATH/../runs/detect/
mkdir -p $LOG_PATH
echo "Logging execution to $LOG_PATH" 
(cd $SCRIPT_PATH/.. && ./src/validation_yolo.py | tee -a $LOG_PATH/validation_yolo.log)
(cd $SCRIPT_PATH/.. && ./src/train_yolo.py | tee -a $LOG_PATH/train_yolo.log)