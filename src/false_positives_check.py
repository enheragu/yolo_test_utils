#!/usr/bin/env python3
# encoding: utf-8

import os

import yaml
from yaml.loader import SafeLoader

import cv2 as cv
import numpy as np

from config import dataset_config_yaml, yolo_dataset_path, yolo_val_output



iou_threshold = 0.5
score_threshold = 0.5


def parseYaml(file_path):
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)


dataset_info = parseYaml(dataset_config_yaml)
color =  [np.random.randint(low=0, high=256, size=3).tolist() for index in range(len(dataset_info['names']))]

# "An Intersection over Union score > 0.5 is normally considered a “good” prediction. "    
def computeIOU(label_orig, label_yolo, eps=1e-7):
    label_orig = [float(i) for i in label_orig]
    label_yolo = [float(i) for i in label_yolo]
    score = label_yolo[5]

    (x1, y1, w1, h1), (x2, y2, w2, h2) = label_orig[1:5], label_yolo[1:5]
    # Taken crom metrics.py of YOLOv8
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

    # Intersection area
    # inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
    #         (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    inter = max(0,(min(b1_x2,b2_x2) - max(b1_x1,b2_x1))) * \
            max(0,(min(b1_y2,b2_y2) - max(b1_y1,b2_y1)))

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    # if iou > 0.7:
    #     print(f"{iou = }")
    
    return iou

def getFalsePositives(original_label_path, detected_label_path):
    false_positives = {}
    labels = []
    labels_detected = []
    labels_original = []

    with open(detected_label_path) as file:
        labels_detected = file.readlines()

    if os.path.exists(original_label_path):
        with open(original_label_path) as file: 
            labels_original = file.readlines()
    # else:
    #     print("Original label path does not exist, everything is a false positive here :)")
    
    for label in labels_detected:
        label = label.strip().split(" ")
        found = False
        if label[0] == '0':  # person
            score = float(label[5])
            # print(f"{label = }")
            for label_orig in labels_original:
                label_orig = label_orig.strip().split(" ")
                if label_orig[0] == '0':  # person
                    iou = computeIOU(label_orig, label)
                    if iou > iou_threshold:
                        found = True
            
            if found and score > score_threshold:
                labels += [label]

    if labels:
        false_positives = {detected_label_path: {'labels': [], 'original': original_label_path}}
        false_positives[detected_label_path]['labels'] = labels

    return false_positives
                
def displayYoloLabel(image, label, dataset_config = dataset_config_yaml):
    label = [float(i) for i in label]
    label[0] = int(label[0])
    data = parseYaml(dataset_config_yaml)
    
    # Start point and endpoint of rectangle
    (x, y, w, h) = label[1:5]
    w_, h_ = w / 2, h / 2
    sp = int((x - w_)*image.shape[1]), int((y - h_)*image.shape[1])
    ep = int((x + w_)*image.shape[0]), int((y + h_)*image.shape[0])


    cv.rectangle(image, sp, ep, color=color[label[0]], thickness=1)

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    label_str = str(data['names'][label[0]])
    (w, h), _ = cv.getTextSize(label_str, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Prints the text.    
    img = cv.rectangle(image, (sp[0], sp[1]-h-8), (sp[0]+w+4, sp[1]), color[label[0]], -1)
    img = cv.putText(image, label_str, (sp[0]+4, int(sp[1]-h/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return img


if __name__ == '__main__':
    
    for root, dirs, files in os.walk(yolo_val_output, topdown=False):
        # When in labels path process all txt
        if 'labels' in root and dirs == []:
            print(f"Process {root} path")
            false_positives = {}
            for file in files:
                if '.txt' not in file:
                    continue
                
                current_test = "/".join(root.replace(yolo_val_output, '').split('/')[1:]).replace("_","/")
                original_label_path = yolo_dataset_path + current_test
                # print(f"{original_label_path = }; {file = }")
                
                false_positives.update(getFalsePositives(os.path.join(original_label_path, file),os.path.join(root, file)))


    for fp_key, fp_values in false_positives.items():
        
        image = cv.imread(fp_values['original'].replace('labels','images').replace('.txt','.png'))
        

        for label in fp_values['labels']:
            image = displayYoloLabel(image, label)
        
        cv.imshow("img_labeled", image)
        key = cv.waitKey(0)

        # check keystroke to exit (image window must be on focus)
        # key = cv.pollKey()
        if key == ord('q') or key == ord('Q') or key == 27:
            break

    cv.destroyAllWindows()