FROM ubuntu:20.04 as base

MAINTAINER Enrique Heredia "e.heredia@umh.es"

# Basic setup of image, update and install utils
# Docker recommends, never put RUN apt-get update alone, causing cache issue ?¿
# RUN apt-get update
RUN apt-get update && apt-get install --fix-missing
# it tries to later install tz data but fails as it has no interactive console, do it manually to setup TZ
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y apt-utils git python3-pip ffmpeg libsm6 libxext6

# Clone and install repository and dependencies
RUN mkdir eeha
RUN cd eeha && git clone --recurse-submodule https://github.com/enheragu/yolo_test_utils
RUN pip install -r eeha/yolo_test_utils/requirements
RUN pip install -r eeha/yolo_test_utils/ultralitics_yolov8/requirements.txt
RUN cd eeha/yolo_test_utils/ultralitics_yolov8 && pip install --editable .

# Creates folder with kaist and runs subfolders to be shared with host when run
RUN mkdir -p eeha/kaist-cvpr15
RUN mkdir -p eeha/kaist-yolo-annotated
RUN mkdir -p eeha/yolo_test_utils/runs

# Ensure git repo is updated every time Docker Image is updated!!
# Change is at the end so not to repeat the whole build process but the last layer
RUN echo "Check updates:"
RUN cd eeha/yolo_test_utils && git pull --recurse-submodules
RUN pip install -r eeha/yolo_test_utils/requirements

ENTRYPOINT ["python3", "eeha/yolo_test_utils/src/run_yolo_test.py"]
CMD ["--help"]