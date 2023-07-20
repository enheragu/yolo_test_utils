FROM ubuntu:20.04 as base

MAINTAINER Enrique Heredia "e.heredia@umh.es"

# Basic setup of image, update and install utils
RUN apt-get update
RUN apt-get install --fix-missing
RUN apt-get install -y apt-utils
RUN apt-get -y install git python3-pip

RUN mkdir -p eeha
RUN cd eeha && git clone --recurse-submodule https://github.com/enheragu/yolo_test_utils
RUN pip install -r eeha/yolo_test_utils/requirements

# Creates folder with kaist and runs subfolders to be shared with host when run
RUN mkdir -p eeha/kaist-cvpr15
RUN mkdir -p eeha/yolo_test_utils/runs

ENTRYPOINT ["python3", "eeha/yolo_test_utils/src/run_yolo_test.py"]