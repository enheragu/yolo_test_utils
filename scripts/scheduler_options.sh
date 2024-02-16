
#!/usr/bin/env bash

export NTFY_TOPIC="eeha_training_test_battery"

## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
SOURCE=$(readlink "$SOURCE")
[[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
export EEHA_SCHEDULER_SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )


function eeha_run_scheduler() {
    ## Activates python vevn and launches script file with provided arguments
    ## Empty arguments will still make use of queued tests :)
    source $EEHA_SCHEDULER_SCRIPT_PATH/../../venv/bin/activate
    (cd $EEHA_SCHEDULER_SCRIPT_PATH/.. && ./src/run_scheduled_test.py $@)
}

function eeha_schedule_new_test() {
    source $EEHA_SCHEDULER_SCRIPT_PATH/../../venv/bin/activate
    (cd $EEHA_SCHEDULER_SCRIPT_PATH/.. && ./src/test_scheduler.py $@)
}

function eeha_stop_scheduler() {
    export EEHA_TEST_STOP_REQUESTED
    # Delete variable -> unset EEHA_TEST_STOP_REQUESTED
}

eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' -rm 'train' --pretrained False \
 --path-name "variance_day_visible_kaist_train" --dataset-format 'kaist' \
 --iterations 2

eeha_run_scheduler