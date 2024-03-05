
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

function eeha_env(){
    source $EEHA_SCHEDULER_SCRIPT_PATH/../../venv/bin/activate
}

function eeha_device0_scheduler() {
    export EEHA_TRAIN_DEVICE=0
    eeha_run_scheduler $@
    #tmux new-session -d -s scheduler_eeha_$EEHA_TRAIN_DEVICE "eeha_run_scheduler $@"
    #tmux detach -s scheduler_eeha_$EEHA_TRAIN_DEVICE
}

function eeha_device1_scheduler() {
    export EEHA_TRAIN_DEVICE=1
    eeha_run_scheduler $@
    #tmux new-session -d -s scheduler_eeha_$EEHA_TRAIN_DEVICE "eeha_run_scheduler $@"
    #tmux detach -s scheduler_eeha_$EEHA_TRAIN_DEVICE
}

function eeha_stop_device0_scheduler() {
    touch $HOME/.cache/eeha_yolo_test/STOP_REQUESTED_GPU0
    # Remove file :) -> rm $HOME/.cache/eeha_yolo_test/STOP_REQUESTED
}

function eeha_stop_device1_scheduler() {
    touch $HOME/.cache/eeha_yolo_test/STOP_REQUESTED_GPU1
    # Remove file :) -> rm $HOME/.cache/eeha_yolo_test/STOP_REQUESTED
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
    touch $HOME/.cache/eeha_yolo_test/STOP_REQUESTED
    # Remove file :) -> rm $HOME/.cache/eeha_yolo_test/STOP_REQUESTED
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

function eeha_print_test_queue() {
    source $EEHA_SCHEDULER_SCRIPT_PATH/../../venv/bin/activate
    (cd $EEHA_SCHEDULER_SCRIPT_PATH/.. && python ./src/check_test_queue.py)
}

function eeha_update_current_cache() {
    echo "Update arvc@10.1.60.87 remote cache data"
    ssh arvc@10.1.60.87 'source ~/eeha/venv/bin/activate; cd ~/eeha/yolo_test_utils/ && python src/GUI/dataset_manager.py;'
    source $EEHA_SCHEDULER_SCRIPT_PATH/../../venv/bin/activate
    echo "Update local cache data"
    (cd $EEHA_SCHEDULER_SCRIPT_PATH/.. && python src/GUI/dataset_manager.py)
    echo "Retrieve remote cache data"
    (cd ~/.cache && scp -r arvc@10.1.60.87:/home/arvc/.cache/eeha_gui_cache .)
}

# eha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 10 --path-name "variance_day_visible_b10_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 20 --path-name "variance_day_visible_b20_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --path-name "variance_day_visible_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 32 --path-name "variance_day_visible_b32_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4x.yaml' --path-name "variance_night_4ch_kaist_trained" --iterations 2
# eeha_schedule_new_test -c 'night' -o 'pca_rgbt_3ch' -m 'yoloCh3x.yaml' --path-name "variance_night_pca_kaist_trained" --iterations 2

# eeha_schedule_new_test -c 'night' -o 'vt' -m 'yoloCh3x.yaml' --path-name "variance_night_vt_kaist_trained" --iterations 2
# eeha_schedule_new_test -c 'night' -o 'lwir' -m 'yoloCh3x.yaml' --path-name "variance_night_lwir_kaist_trained" --iterations 2
# eeha_schedule_new_test -c 'day' -o 'vt' -m 'yoloCh3x.yaml' --path-name "variance_day_vt_kaist_trained" --iterations 2
# eeha_schedule_new_test -c 'day' -o 'hsvt' -m 'yoloCh3x.yaml' --path-name "variance_day_hsvt_kaist_trained" --iterations 2

# eeha_run_scheduler



## Schedule in 3090
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 20 --path-name "variance_day_visible_b20_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 20 --path-name "variance_day_visible_b20_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --path-name "variance_day_visible_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --path-name "variance_day_visible_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 5 --path-name "variance_day_visible_b5_kaist_trained_GPU3090" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 5 --path-name "variance_day_visible_b5_kaist_trained_GPU3090" --iterations 5

# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 20 --path-name "variance_day_visible_b20_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 20 --path-name "variance_day_visible_b20_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --path-name "variance_day_visible_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --path-name "variance_day_visible_kaist_trained" --iterations 4
# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 5 --path-name "variance_day_visible_b5_kaist_trained" --iterations 5
# eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 5 --path-name "variance_day_visible_b5_kaist_trained" --iterations 5


## Schedule in 4090
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --deterministic False --path-name "variance_day_visible_kaist_trained_nondet" --iterations 2
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 20 --path-name "variance_day_visible_b20_kaist_trained" --iterations 3
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 5 --path-name "variance_day_visible_b5_kaist_trained" --iterations 10
# EXEC # eeha_schedule_new_test -c 'day' -o 'hsvt' -m 'yoloCh3x.yaml' --path-name "variance_day_hsvt_kaist_trained" --iterations 9
# EXEC # eeha_schedule_new_test -c 'day' -o 'vt' -m 'yoloCh3x.yaml' --path-name "variance_day_vt_kaist_trained" --iterations 6
# EXEC # eeha_schedule_new_test -c 'day' -o 'vt' -m 'yoloCh3x.yaml' --path-name "variance_day_vt_kaist_trained" --iterations 6
