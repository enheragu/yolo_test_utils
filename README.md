# yolo_test_utils

Collection of helper python scripts to run different tests with YOLO along with dataset manipulation.

· src/simple_test.py -> Runs YOLO validation test based on a set of models an set of datasets.
· src/kaist_to_yolo_annotations.py -> Creates a new folder with the Kaist dataset ordered and labelled so to be used with YOLO.
· src/kaist_image_label.py -> Visually recreates kaist dataset drawing rectangle and class over each annotation.
· src/gather_results.py -> Gathers results from simple_test into a table an composed precission-recall graphs.