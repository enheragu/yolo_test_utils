FROM ubuntu:20.04 as base

MAINTAINER Enrique Heredia "e.heredia@umh.es"

# Basic setup of image, update and install utils
# Docker recommends, never put RUN apt-get update alone, causing cache issue ?¿
# RUN apt-get update
RUN apt-get update && apt-get install --fix-missing
# it tries to later install tz data but fails as it has no interactive console, do it manually to setup TZ
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y apt-utils git python3-pip ffmpeg libsm6 libxext6 coreutils

# Clone and install repository and dependencies
RUN mkdir root/eeha
RUN cd root/eeha && git clone --recurse-submodule https://github.com/enheragu/yolo_test_utils
RUN pip install -r root/eeha/yolo_test_utils/requirements
RUN pip install -r root/eeha/yolo_test_utils/ultralitics_yolov8/requirements.txt
RUN cd root/eeha/yolo_test_utils/ultralitics_yolov8 && pip install --editable .

# Creates folder with kaist and runs subfolders to be shared with host when run
RUN mkdir -p root/eeha/kaist-cvpr15
RUN mkdir -p root/eeha/kaist-yolo-annotated
RUN mkdir -p root/eeha/yolo_test_utils/runs

# Ensure git repo is updated every time Docker Image is updated!!
# Change is at the end so not to repeat the whole build process but the last layer
RUN echo "Check updates: 2"
RUN cd root/eeha/yolo_test_utils && git pull --recurse-submodules
RUN pip install -r root/eeha/yolo_test_utils/requirements

# Setup workdir so script is executed from correct path
# WORKDIR /root/root/eeha/yolo_test_utils

# Generate entrypoint with bash at the end to have interactive mode in case it fails
RUN echo '# entrypoint.sh\n#!/usr/bin/env bash\n(cd /root/eeha/yolo_test_utils && python3 ./src/run_yolo_test.py "$@" 2>&1 | tee -a /root/eeha/yolo_test_utils/runs/docker_test_out.log); /bin/bash' > /root/eeha/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/root/eeha/entrypoint.sh"]
CMD ["--help"]