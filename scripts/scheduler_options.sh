
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

    tail -f $EEHA_SCHEDULER_SCRIPT_PATH/../runs/now_executing${tag}.log -n 500
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
# rsync -avz -e "ssh" --exclude='*.jpg' --exclude='*.png' --exclude='*.tar.gz' --exclude='*.pt' --exclude='*/labels/' \
#   arvc@arvc-gpu:/home/arvc/eeha/kaist-cvpr15/runs/detect/ \
#   arvc@arvc-gpu:/home/arvc/eeha/yolo_test_utils/runs/detect/ \
#   ./detect


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
# EXEC # eeha_schedule_new_test -c 'night' -o 'pca' -m 'yoloCh3x.yaml' --path-name "variance_night_pca_kaist_trained" --cache "disk" --iterations 5

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


## Add rgbt taking t as alpha channel in PNG and converting to BGR directly. Execute all options with and without equalization
# eeha_schedule_new_test -c 'day' 'night' -o 'alphat_rgbt' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_sameseed"
# eeha_schedule_new_test -c 'day' 'night' -o 'alphat_rgbt' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "rgb_th_equalization_sameseed" --th_equalization 'clahe' --rgb_equalization 'clahe'
# eeha_schedule_new_test -c 'day' 'night' -o 'alphat_rgbt' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "rgb_equalization_sameseed" --rgb_equalization 'clahe'
# eeha_schedule_new_test -c 'day' 'night' -o 'alphat_rgbt' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "th_equalization_sameseed" --th_equalization 'clahe'



## KAIST CORRECTED

# Reference without correction and relabeling
# eeha_schedule_new_test -c 'day' 'night' -o '4ch' -m 'yoloCh4m.yaml' --dataset-format "kaist_80_20" --cache "disk" --distortion_correct "False" --relabeling "False" --path-name "no_equalization_sameseed"
# Relabeled
# eeha_schedule_new_test -c 'day' 'night' -o 'visible' 'lwir' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_sameseed_corrected_relabeled"
# eeha_schedule_new_test -c 'day' 'night' -o 'alphat_rgbt' 'hsvt' 'vt' 'rgbt_v2' 'vths_v2'   -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_sameseed_corrected_relabeled"
# eeha_schedule_new_test -c 'day' 'night' -o '4ch' -m 'yoloCh4m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "no_equalization_sameseed_corrected_relabeled"

# Optical flow
# eeha_schedule_new_test -c 'night' -o 'alphat_rgbt' 'lwir' 'visible'  -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_sameseed_opticalflow_v2"


## MDPI first review results 
# Kaist
# eeha_schedule_new_test -c 'day' 'night' -o 'visible' 'lwir' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_mdpi_review"
# eeha_schedule_new_test -c 'day' 'night' -o 'hsvt' 'vt' 'rgbt_v2' 'vths_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_mdpi_review"
# (both eq)
# eeha_schedule_new_test -c 'day' 'night' -o 'visible' 'lwir' 'hsvt' 'vt' 'rgbt_v2' 'vths_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "rgb_th_equalization_mdpi_review" --th_equalization 'clahe' --rgb_equalization 'clahe'
# (rgb_eq)
# eeha_schedule_new_test -c 'day' 'night' -o 'hsvt' 'vt' 'rgbt_v2' 'vths_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "rgb_equalization_mdpi_review" --rgb_equalization 'clahe'
# (th_eq)
# eeha_schedule_new_test -c 'day' 'night' -o 'hsvt' 'vt' 'rgbt_v2' 'vths_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "th_equalization_mdpi_review" --th_equalization 'clahe'
# (4ch)
# eeha_schedule_new_test -c 'day' 'night' -o '4ch' -m 'yoloCh4_v2m.yaml' --test-name "yoloCh4v2" --dataset-format "kaist_80_20" --cache "disk" --path-name "no_equalization_mdpi_review"
# eeha_schedule_new_test -c 'day' 'night' -o '4ch' -m 'yoloCh4_v3m.yaml' --test-name "yoloCh4v3" --dataset-format "kaist_80_20" --cache "disk" --path-name "no_equalization_mdpi_review"
# eeha_schedule_new_test -c 'day' 'night' -o '4ch' -m 'yoloCh4m.yaml' --test-name "yoloCh4" --dataset-format "kaist_80_20" --cache "disk" --path-name "no_equalization_mdpi_review"

# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_v2m.yaml' --test-name "yoloCh4v2" --dataset-format "kaist_80_20" --cache "disk" --path-name "no_equalization_mdpi_review" --distortion_correct "False" --relabeling "False"
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_v3m.yaml' --test-name "yoloCh4v3" --dataset-format "kaist_80_20" --cache "disk" --path-name "no_equalization_mdpi_review" --distortion_correct "False" --relabeling "False"
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4m.yaml' --test-name "yoloCh4" --dataset-format "kaist_80_20" --cache "disk" --path-name "no_equalization_mdpi_review" --distortion_correct "False" --relabeling "False"
# LLVIP
# eeha_schedule_new_test -c 'night' -o 'visible' 'lwir' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_no_equalization_mdpi_review"
# eeha_schedule_new_test -c 'night' -o 'hsvt' 'vt' 'rgbt_v2' 'vths_v2' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_no_equalization_mdpi_review"
# (both eq)
# eeha_schedule_new_test -c 'night' -o 'visible' 'lwir' 'hsvt' 'vt' 'rgbt_v2' 'vths_v2' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_rgb_th_equalization_mdpi_review" --th_equalization 'clahe' --rgb_equalization 'clahe'
# (rgb_eq)
# eeha_schedule_new_test -c 'night' -o 'hsvt' 'vt' 'rgbt_v2' 'vths_v2' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_rgb_equalization_mdpi_review" --rgb_equalization 'clahe'
# (th_eq)
# eeha_schedule_new_test -c 'night' -o 'hsvt' 'vt' 'rgbt_v2' 'vths_v2' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_th_equalization_mdpi_review" --th_equalization 'clahe'
# (4ch)
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_v2m.yaml' --test-name "yoloCh4v2" --dataset-format "llvip_80_20" --cache "disk" --path-name "llvip_no_equalization_mdpi_review"
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_v3m.yaml' --test-name "yoloCh4v3" --dataset-format "llvip_80_20" --cache "disk" --path-name "llvip_no_equalization_mdpi_review"
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4m.yaml' --test-name "yoloCh4" --dataset-format "llvip_80_20" --cache "disk" --path-name "llvip_no_equalization_mdpi_review" 

## Test label-paired version
# eeha_schedule_new_test -c 'night' -o 'rgbt_v2' 'vths_v2'  -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_labels_paired"
# eeha_schedule_new_test -c 'day' -o 'rgbt_v2' --test-name 'day_rgbt_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_labels_paired"  --distortion_correct "False" --relabeling "False"
# eeha_schedule_new_test -c 'day' -o 'rgbt_v2' --test-name 'day_rgbt_v2_corrected' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_labels_paired"
# eeha_schedule_new_test -c 'day' -o 'hsvt' --test-name 'day_hsvt' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_labels_paired"  --distortion_correct "False" --relabeling "False"
# eeha_schedule_new_test -c 'day' -o 'hsvt' --test-name 'day_hsvt_corrected' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "no_equalization_labels_paired"
# eeha_schedule_new_test -c 'day' -o '4ch' -m 'yoloCh4m.yaml' --test-name "day_yoloCh4m" --dataset-format "kaist_80_20" --cache "disk" --path-name "no_equalization_labels_paired"
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4m.yaml' --test-name "night_yoloCh4m" --dataset-format "kaist_80_20" --cache "disk" --path-name "no_equalization_labels_paired"

###################################
##       NON-STATIC PAPER        ##
###################################


# LLVIP
# VARIANCE SHORT 
# (4ch early fusion)
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_early_v1m.yaml' --test-name "Ch4_early_v1" --dataset-format "llvip_80_20" --path-name "variance_llvip_Ch4_early_v1" --iterations 3 --cache "disk"
# TEST # eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_early_v1m.yaml' --test-name "Ch4_early_v1" --dataset-format "llvip_80_20" --path-name "0_tmp_early" --iterations 1 --cache "disk"
# (4ch middle fusion)
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_middle_v1m.yaml' --test-name "Ch4_middle_v1" --dataset-format "llvip_80_20" --path-name "variance_llvip_Ch4_middle_v1" --iterations 3 --cache "disk"
# (4ch late fusion)
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_late_v1m.yaml' --test-name "Ch4_late_v1" --dataset-format "llvip_80_20" --path-name "variance_llvip_Ch4_late_v1" --iterations 3 --cache "disk"
# (4ch split late fusion)
# eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_late_split_v1m.yaml' --test-name "Ch4_late_split_v1" --dataset-format "llvip_4ch_latesplit_80_20" --path-name "variance_llvip_Ch4_late_split_v1" --iterations 3 --cache "disk"
# TEST # eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_late_split_v1m.yaml' --test-name "Ch4_late_split_v1" --dataset-format "llvip_4ch_latesplit_80_20" --path-name "0_tmp_late_split_test" --iterations 1 --cache "disk"

# eeha_schedule_new_test -c 'night' -o 'visible' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_visible" --iterations 3
# eeha_schedule_new_test -c 'night' -o 'lwir' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_lwir" --iterations 3

## TO ICINCO
# eeha_schedule_new_test -c 'night' -o 'pca' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_pca" --iterations 3  --cache "disk"
# eeha_schedule_new_test -c 'night' -o 'fa' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_fa_patches" --iterations 3  --cache "disk"
# eeha_schedule_new_test -c 'night' -o 'wavelet' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_wavelet" --iterations 3  --cache "disk"
# eeha_schedule_new_test -c 'night' -o 'curvelet' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_curvelet" --iterations 3  --cache "disk"
# eeha_schedule_new_test -c 'night' -o 'wavelet_max' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_wavelet_max" --iterations 3  --cache "disk"
# eeha_schedule_new_test -c 'night' -o 'curvelet_max' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_curvelet_max" --iterations 3  --cache "disk"

# eeha_schedule_new_test -c 'night' -o 'alpha_pca' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_alpha_pca" --iterations 3  --cache "disk"
# eeha_schedule_new_test -c 'night' -o 'ssim' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_ssim" --iterations 2
# eeha_schedule_new_test -c 'night' -o 'ssim_v2' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_ssim_v2" --iterations 2
# eeha_schedule_new_test -c 'night' -o 'superpixel' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_superpixel" --iterations 3
# eeha_schedule_new_test -c 'night' -o 'sobel_weighted' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_sobel_weighted" --iterations 3

# NEED CACHED DISK # eeha_schedule_new_test -c 'night' -o 'pca' 'fa' 'wavelet' 'curvelet' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_static_v2_no_equalization"  --cache "disk"
# NO NEED CACHED   # eeha_schedule_new_test -c 'night' -o 'ssim' 'superpixel' 'sobel_weighted' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_static_v2_no_equalization"

# (both eq)
# NEED CACHED DISK # eeha_schedule_new_test -c 'night' -o 'pca' 'fa' 'wavelet' 'curvelet' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_static_v2_rgb_th_equalization" --th_equalization 'clahe' --rgb_equalization 'clahe'  --cache "disk"
# NO NEED CACHED   # eeha_schedule_new_test -c 'night' -o 'ssim' 'superpixel' 'sobel_weighted' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_static_v2_rgb_th_equalization" --th_equalization 'clahe' --rgb_equalization 'clahe'  --cache "disk"
# (rgb_eq)
# NEED CACHED DISK # eeha_schedule_new_test -c 'night' -o 'pca' 'fa' 'wavelet' 'curvelet' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_static_v2_rgb_equalization" --rgb_equalization 'clahe'  --cache "disk"
# NO NEED CACHED   # eeha_schedule_new_test -c 'night' -o 'ssim' 'superpixel' 'sobel_weighted' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_static_v2_rgb_equalization" --rgb_equalization 'clahe'  --cache "disk"
# (th_eq)
# NEED CACHED DISK # eeha_schedule_new_test -c 'night' -o 'pca' 'fa' 'wavelet' 'curvelet' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_static_v2_th_equalization" --th_equalization 'clahe'  --cache "disk"
# NO NEED CACHED   # eeha_schedule_new_test -c 'night' -o 'ssim' 'superpixel' 'sobel_weighted' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "llvip_static_v2_th_equalization" --th_equalization 'clahe'  --cache "disk"


## First channel is the only one with information :(
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca' 'fa' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_no_equalization"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca' 'fa' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_th_equalization" --th_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca' 'fa' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "rgb_equalization_sameseed" --rgb_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca' 'fa' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_rgb_th_equalization" --th_equalization 'clahe' --rgb_equalization 'clahe'

## Triplicates first component to work with one channel but with normal model
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_1ch' 'fa_rgbt_1ch' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_no_equalization"
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_1ch' 'fa_rgbt_1ch' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_th_equalization" --th_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_1ch' 'fa_rgbt_1ch' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "rgb_equalization_sameseed" --rgb_equalization 'clahe'
# EXEC # eeha_schedule_new_test -c 'day' 'night' -o 'pca_rgbt_1ch' 'fa_rgbt_1ch' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --cache "disk" --path-name "nonstatic_rgb_th_equalization" --th_equalization 'clahe' --rgb_equalization 'clahe'

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
# EXEC # eeha_schedule_new_test -c 'day' -o 'visible' -m 'yoloCh3m.yaml' --deterministic False --dataset-format "coco" --path-name "variance_default_coco"  --iterations 2

## Variance RGBT with KAIST
# EXEC # eeha_schedule_new_test -c 'day' -o 'rgbt_v2' -m 'yoloCh3m.yaml' --dataset-format "kaist_80_20" --path-name "variance_day_rgbt" --iterations 10
## Variance VT (th eq) with LLVIP
# EXEC # eeha_schedule_new_test -c 'night' -o 'vt' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_night_vt" --th_equalization 'clahe' --iterations 10
# EXEC # eeha_schedule_new_test -c 'night' -o '4ch' -m 'yoloCh4_v3m.yaml' --dataset-format "llvip_80_20"  --test-name "yoloCh4v3" --cache "disk" --path-name "variance_llvip_night_yoloCh4v3" --iterations 10
# EXEC # eeha_schedule_new_test -c 'night' -o 'lwir' -m 'yoloCh3m.yaml' --dataset-format "llvip_80_20" --path-name "variance_llvip_night_lwir" --iterations 10