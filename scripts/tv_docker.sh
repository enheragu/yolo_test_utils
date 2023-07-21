



# Need to share paths as volumes in Docker container
export DATASET_ORIGINAL_PATH=${HOME}/eeha/kaist-cvpr15
export DATASET_ANNOTATED_PATH=${HOME}/eeha/kaist-yolo-annotated
export RUN_TEST_PATH=${HOME}/eeha/yolo_test_utils/runs

# Without sudo does not recognice GPU libs?¿
sudo docker run -it \
    --gpus all \
    --volume="$DATASET_ORIGINAL_PATH:/root/eeha/kaist-cvpr15" \
    --volume="$DATASET_ANNOTATED_PATH:/root/eeha/kaist-yolo-annotated" \
    --volume="$RUN_TEST_PATH:/root/eeha/yolo_test_utils/runs" \
    enheragu/yolo_tests -o 'visible' 'lwir' 'hsvt' 'rgbt' 'vths' 'vt' -c 'day' 'night' -d '0' -rm 'val' 'train' -m yolov8x.pt