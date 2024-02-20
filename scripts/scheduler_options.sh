
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


function eeha_device0_scheduler() {
    export EEHA_TRAIN_DEVICE=0
    tmux new-session -d -s scheduler_$EEHA_TRAIN_DEVICE "eeha_run_scheduler $@"
    tmux detach -s scheduler_$EEHA_TRAIN_DEVICE
}

function eeha_device1_scheduler() {
    export EEHA_TRAIN_DEVICE=1
    tmux new-session -d -s scheduler_$EEHA_TRAIN_DEVICE "eeha_run_scheduler $@"
    tmux detach -s scheduler_$EEHA_TRAIN_DEVICE
}

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

function eeha_run_gui() {
    source $EEHA_SCHEDULER_SCRIPT_PATH/../../venv/bin/activate
    (cd $EEHA_SCHEDULER_SCRIPT_PATH/.. && ./src/gui_plot_results.py $@ &)
}

function eeha_stop_gui() {
    kill $(ps -aux | grep "gui_plot_results" | awk '{print $2}')
}

function eeha_check_process() {
    tail -f $EEHA_SCHEDULER_SCRIPT_PATH/../runs/now_executing.log -n 300
}


# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 20 --path-name "variance_day_visible_b20_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 32 --path-name "variance_day_visible_b32_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4x.yaml' --path-name "variance_night_4ch_kaist_trained" --iterations 2
# eeha_schedule_new_test -c 'night' -o 'pca_rgbt_3ch' -m 'yoloCh3x.yaml' --path-name "variance_night_pca_kaist_trained" --iterations 2

# eeha_schedule_new_test -c 'night' -o 'vt' -m 'yoloCh3x.yaml' --path-name "variance_night_vt_kaist_trained" --iterations 2
# eeha_schedule_new_test -c 'night' -o 'lwir' -m 'yoloCh3x.yaml' --path-name "variance_night_lwir_kaist_trained" --iterations 2
# eeha_schedule_new_test -c 'day' -o 'vt' -m 'yoloCh3x.yaml' --path-name "variance_day_vt_kaist_trained" --iterations 2
# eeha_schedule_new_test -c 'day' -o 'hsvt' -m 'yoloCh3x.yaml' --path-name "variance_day_hsvt_kaist_trained" --iterations 2

# eeha_run_scheduler