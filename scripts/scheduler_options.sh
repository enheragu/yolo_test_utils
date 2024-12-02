
#!/usr/bin/env bash

export NTFY_TOPIC="eeha_training_test_battery"

## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
SOURCE=$(readlink "$SOURCE")
[[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
export EEHA_SCHEDULER_SCRIPTFILE_PATH=$SOURCE
export EEHA_SCHEDULER_SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

function eeha_env(){
    source $EEHA_SCHEDULER_SCRIPT_PATH/../../venv/bin/activate
}

function eeha_device0_scheduler() {
    # export EEHA_ACTIVE_TEST_TIMETABLE='15:30-06:00'
    # Propagate options ti child
    tmux new-session -d -s "scheduler_eeha_0" "export EEHA_TRAIN_DEVICE_TAG=$EEHA_TRAIN_DEVICE_TAG;  
                                               export EEHA_ACTIVE_TEST_TIMETABLE=$EEHA_ACTIVE_TEST_TIMETABLE; 
                                               export EEHA_TRAIN_DEVICE=0; 
                                               source $EEHA_SCHEDULER_SCRIPTFILE_PATH; 
                                               eeha_run_scheduler $@;"
}

function eeha_device1_scheduler() {
    tmux new-session -d -s "scheduler_eeha_1" "export EEHA_TRAIN_DEVICE_TAG=$EEHA_TRAIN_DEVICE_TAG;  
                                               export EEHA_ACTIVE_TEST_TIMETABLE=$EEHA_ACTIVE_TEST_TIMETABLE; 
                                               export EEHA_TRAIN_DEVICE=1; 
                                               source $EEHA_SCHEDULER_SCRIPTFILE_PATH; 
                                               eeha_run_scheduler $@;"
}

function eeha_stop_device0_scheduler() {
    touch $HOME/.cache/eeha_yolo_test/STOP_REQUESTED_GPU0
    # Remove file :) -> rm $HOME/.cache/eeha_yolo_test/STOP_REQUESTED
}

function eeha_stop_device1_scheduler() {
    touch $HOME/.cache/eeha_yolo_test/STOP_REQUESTED_GPU1
    # Remove file :) -> rm $HOME/.cache/eeha_yolo_test/STOP_REQUESTED
}



function eeha_keep_scheduler_running()
{
    export PENDING_TESTS="/home/arvc/.cache/eeha_yolo_test/pending.yaml"
    export SCHEDULER_CMD="python3 ./src/run_scheduled_test.py"

    while true; do
        # Check if the process is running
        if pgrep -f "$SCHEDULER_CMD" > /dev/null; then
            echo "Scheduler process is still running :) just do nothing, checking again in 30 min."
            sleep 900  # sleep for 30 minutes
        else
            eeha_run_scheduler
        fi

        # Check if the YAML file exists and has relevant content
        if [ -f "$PENDING_TESTS" ]; then
            if grep -q '^[^#[:space:]]' "$PENDING_TESTS" && ! grep -q '^\[\]$' "$PENDING_TESTS"; then
                echo "YAML file ($PENDING_TESTS) still has content, keeping the loop."
            else
                echo "YAML file ($PENDING_TESTS) is empty, no more tests to execute :)"
                break
            fi
        else
            echo "YAML file ($PENDING_TESTS) does not exist, exiting."
            break
        fi
    done
}


function eeha_run_scheduler() {
    echo "EEHA_ACTIVE_TEST_TIMETABLE: $EEHA_ACTIVE_TEST_TIMETABLE"
    echo "EEHA_TRAIN_DEVICE_TAG: $EEHA_TRAIN_DEVICE_TAG"

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

function eeha_kill_scheduler() {
    kill $(ps -aux | grep "run_scheduled_test.py" | awk '{print $2}')
}

function eeha_run_gui() {
    source $EEHA_SCHEDULER_SCRIPT_PATH/../../venv/bin/activate
    (cd $EEHA_SCHEDULER_SCRIPT_PATH/.. && ./src/gui_plot_results.py $@ &)
}

function eeha_kill_gui() {
    kill $(ps -aux | grep "gui_plot_results" | awk '{print $2}')
}


function eeha_check_process() {
    tag=""

    if [[ -v EEHA_TRAIN_DEVICE_TAG ]]; then
        tag+="_${EEHA_TRAIN_DEVICE_TAG}"
    fi

    if [[ -v EEHA_TRAIN_DEVICE ]]; then
        tag+="_GPU${EEHA_TRAIN_DEVICE}"
    fi

    tail -f $EEHA_SCHEDULER_SCRIPT_PATH/../runs/now_executing${tag}.log -n 300
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

function rsync_cache() {
    rsync -avz --delete arvc@arvc-gpu:/home/arvc/.cache/eeha_gui_cache ~/.cache/
}

###############################
##  VARIANCE ANALYSIS TESTS  ##
###############################

### 4090 test variance set
# EXEC # eeha_schedule_new_test -c 'night' -o 'vt' -m 'yoloCh3x.yaml' --path-name "variance_night_vt_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'vt' -m 'yoloCh3x.yaml' --path-name "variance_day_vt_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'hsvt' -m 'yoloCh3x.yaml' --path-name "variance_day_hsvt_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'night' -o 'vt_2ch' -m 'yoloCh2x.yaml' --path-name "variance_night_vtch2_kaist_trained" --cache "disk" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'vths' -m 'yoloCh3x.yaml' --path-name "variance_day_vths_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4x.yaml' --path-name "variance_night_4ch_kaist_trained" --cache "disk" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o '4ch' -m 'yoloCh4x.yaml' --path-name "variance_day_4ch_kaist_trained" --cache "disk" --iterations 5
# EXEC # eeha_schedule_new_test -c 'night' -o 'rgbt' -m 'yoloCh3x.yaml' --path-name "variance_night_rgbt_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'rgbt' -m 'yoloCh3x.yaml' --path-name "variance_day_rgbt_kaist_trained" --iterations 5
# EXEC # eeha_schedule_new_test -c 'night' -o 'pca_rgbt_npy' -m 'yoloCh3x.yaml' --path-name "variance_night_pca_kaist_trained" --cache "disk" --iterations 5

### HIPERPARAM TESTS
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 5 --path-name "variance_day_visible_b5_kaist_trained" --iterations 10
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 10 --path-name "variance_day_visible_b10_kaist_trained" --iterations 10
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --batch 20 --path-name "variance_day_visible_b20_kaist_trained" --iterations 10
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --path-name "variance_day_visible_kaist_trained" --iterations 10
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3x.yaml' --deterministic False --path-name "variance_day_visible_kaist_nondet_trained" --iterations 10

## PRETRAINED TESTS
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yolov8x.pt' --dataset-format 'kaist_coco' --pretrained True --path-name "variance_day_visible_pretrained" --iterations 10



#####################################
##  BADAJOZ STATIC PAPER SIMPOSIO  ##
#####################################

# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'visible' -m 'yoloCh3x.yaml' --dataset-format "kaist_full"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'lwir' -m 'yoloCh3x.yaml' --dataset-format "kaist_full"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'hsvt' -m 'yoloCh3x.yaml' --dataset-format "kaist_full"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'rgbt' -m 'yoloCh3x.yaml' --dataset-format "kaist_full"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'vt' -m 'yoloCh3x.yaml' --dataset-format "kaist_full"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'vths' -m 'yoloCh3x.yaml' --dataset-format "kaist_full"


#######################################
##  BADAJOZ V2 STATIC PAPER REVISTA  ##
#######################################
# CLAHE for LWIR image an better splitting of datasets. Compare against 'all' together


## Equalization ablation.

# hsvt     ->  average
# rgbt     ->  average
# vths     ->  bit sifting
# vths_v2  ->  /16 and merge
# vths_v3  ->  average
# vt       ->  average

## From no trained model!!!
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'visible' 'lwir' 'hsvt' 'vt' 'vths_v2' 'rgbt_v2' -m 'yoloNoTrained.pt' --dataset-format "kaist_80_20" --path-name "rgb_th_equalization_notrained" --th_equalization 'clahe' --rgb_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'hsvt' 'vt' 'vths_v2' 'rgbt_v2' -m 'yoloNoTrained.pt' --dataset-format "kaist_80_20" --path-name "rgb_equalization_notrained" --rgb_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'hsvt' 'vt' 'vths_v2' 'rgbt_v2' -m 'yoloNoTrained.pt' --dataset-format "kaist_80_20" --path-name "th_equalization_notrained" --th_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'visible' 'lwir' 'hsvt' 'vt' 'vths_v2' 'rgbt_v2' -m 'yoloNoTrained.pt' --dataset-format "kaist_80_20" --path-name "no_equalization_notrained"
###

# Check two best (split in two to save space) 
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'vths' 'vt' 'rgbt' 'hsvt' 'vths_v2' 'vths_v3' 'rgbt_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_sameseed"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'visible' 'lwir' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_sameseed"

# Effects of equalization on original images 
# (both eq)
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'hsvt' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "rgb_th_equalization_sameseed" --th_equalization 'clahe' --rgb_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'visible' 'lwir' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "rgb_th_equalization_sameseed" --th_equalization 'clahe' --rgb_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'rgbt' 'vt' 'vths_v2' 'vths_v3' 'rgbt_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "rgb_th_equalization_sameseed" --th_equalization 'clahe' --rgb_equalization 'clahe'

# (rgb_eq)
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'hsvt' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "rgb_equalization_sameseed" --rgb_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'rgbt' 'vt' 'vths_v2' 'vths_v3' 'rgbt_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "rgb_equalization_sameseed" --rgb_equalization 'clahe'

# (th_eq)
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'hsvt' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "th_equalization_sameseed" --th_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'rgbt' 'vt' 'vths_v2' 'vths_v3' 'rgbt_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "th_equalization_sameseed" --th_equalization 'clahe'


# Effects of dataset distribution of the original images
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'visible' 'lwir' -m 'yoloCh3m.yaml' --dataset-format "kaist_70_30"  --path-name "kaist_70_30"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'visible' 'lwir' -m 'yoloCh3m.yaml' --dataset-format "kaist_90_10" --path-name "kaist_90_10"

# Is sameseed really working?
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "variance_sameseed" --iterations 5
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloNoTrained.pt' --dataset-format "kaist_80_20" --path-name "variance_notrained" --iterations 5


###################################
##       NON-STATIC PAPER        ##
###################################

## First channel is the only one with information :(
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_npy' 'fa_rgbt_npy' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_no_equalization"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_npy' 'fa_rgbt_npy' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_th_equalization" --th_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_npy' 'fa_rgbt_npy' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "rgb_equalization_sameseed" --rgb_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_npy' 'fa_rgbt_npy' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_rgb_th_equalization" --th_equalization 'clahe' --rgb_equalization 'clahe'

## Triplicates first component to work with one channel but with normal model
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_1ch_npy' 'fa_rgbt_1ch_npy' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_no_equalization"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_1ch_npy' 'fa_rgbt_1ch_npy' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_th_equalization" --th_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_1ch_npy' 'fa_rgbt_1ch_npy' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "rgb_equalization_sameseed" --rgb_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_1ch_npy' 'fa_rgbt_1ch_npy' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_rgb_th_equalization" --th_equalization 'clahe' --rgb_equalization 'clahe'

# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'lwir_1ch' -m 'yoloCh1x.yaml' --dataset-format "kaist_full" --cache "disk"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'vt_2ch' -m 'yoloCh2x.yaml' --dataset-format "kaist_full" --cache "disk"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o '4ch' -m 'yoloCh4x.yaml' --dataset-format "kaist_full" --cache "disk"
 


#####################
##  DEBUG TEST :)  ##
#####################
# DEBUG #eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3m.yaml' --dataset-format "kaist_debug" --path-name "tmp_debug_to_delete"



#######################################
##  Variance COCO and modifications  ##
#######################################

# Run variance training set with YOLO and COCO dataset
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3m.yaml' --batch 32 --dataset-format "coco" --path-name "variance_default_coco" --iterations 5