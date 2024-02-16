import os
import sys

ultralitics_rel_path = (os.path.dirname(os.path.abspath(__file__)) + '/../ultralitics_yolov8/ultralytics/',
                        os.path.dirname(os.path.abspath(__file__)) + '/../ultralitics_yolov8/',
                        os.path.dirname(os.path.abspath(__file__)) + '/../'
                        )
yolo_test_paths = (os.path.dirname(os.path.abspath(__file__)) + '/Dataset',
                   os.path.dirname(os.path.abspath(__file__)) + '/YoloExecution'
                  )

for add_path in ultralitics_rel_path + yolo_test_paths:
    try:
        sys.path.index(add_path) # Or os.getcwd() for this directory
    except ValueError:
        sys.path.append(add_path) # Or os.getcwd() for this directory
