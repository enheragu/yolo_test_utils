
import os
import cv2
import re
import numpy as np
from itertools import product
import pickle
from pathlib import Path

from utils import log, bcolors
from Dataset.constants import kaist_path, images_folder_name, labels_folder_name

image_shape = (640,512)

optical_flow_cache = {}
def load_transform(image_path, optical_flow_data_path = '/home/arvc/eeha/multiespectral_correction/data'):
    global optical_flow_cache
    pattern = r"(set\d{2}).*?(V\d{3})"
    match = re.search(pattern, image_path)
    set_name = match.group(1)
    sequence_name = match.group(2)
    if f"{set_name}/{sequence_name}" not in optical_flow_cache:
        optical_flow_file = f'{optical_flow_data_path}/transform_{set_name}_{sequence_name}.pkl'

        with open(optical_flow_file, 'rb') as file:
            data = pickle.load(file)
        optical_flow_cache[f"{set_name}/{sequence_name}"] = data
    optical_flow_list = optical_flow_cache[f"{set_name}/{sequence_name}"]
    if not image_path in optical_flow_list:
        return np.eye(2, 3)
    else:
        return optical_flow_list[image_path]
    
    



crop_x, crop_y, crop_w, crop_h = 80, 30, 550, 440
crop_data = crop_x, crop_y, crop_w, crop_h

camera_matrix_est = np.array([[2.25274009e+03, 0.00000000e+00, 1.87117718e+02],
                              [0.00000000e+00, 1.95608933e+03, 1.41560811e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_coeffs_est = np.array([[-2.83825789e+00],
                            [ 1.26546508e+01],
                            [ 2.49520747e-02],
                            [ 9.63645328e-04],
                            [-5.66839388e+00]])

def fillBorders(image, x, y, w, h):
        cropped_image = image[y:y+h, x:x+w]

        mosaic_image = np.zeros_like(image)
        mosaic_image[:h, :w] = cropped_image
        mosaic_image[:h, w:] = cropped_image[:, :image.shape[1]-w]
        mosaic_image[h:, :w] = cropped_image[:image.shape[0]-h, :]
        mosaic_image[h:, w:] = cropped_image[:image.shape[0]-h, :image.shape[1]-w]
    
        return mosaic_image

def generateCorrectedImage(image, data_type, image_path, camera_matrix_est = camera_matrix_est, dist_coeffs_est = dist_coeffs_est, crop_data = crop_data):
    crop_x, crop_y, crop_w, crop_h = crop_data

    image_data_tag = image_path.replace('lwir','visible').replace(os.path.join(kaist_path,images_folder_name),'')
    tranform_mat = load_transform(image_data_tag)
    rows, cols = image.shape[:2]
    corrected_image = cv2.warpAffine(image, tranform_mat, (cols, rows))

    corrected_image = fillBorders(corrected_image, crop_x, crop_y, crop_w, crop_h)
    return corrected_image

    
def generateCorrectedLabels(label, xml_path, crop_data = crop_data, image_shape = image_shape, labeled=None):
    epsilon_pixels = 3
    crop_x, crop_y, crop_w, crop_h = crop_data
    img_w, img_h = image_shape

    corrected_labels = []
    obj_id, x, y, w, h = label
    
    cx = x - w/2
    cy = y - h/2
    # If center of obj is outside of image class is not tagged
    if cx < epsilon_pixels or cy < epsilon_pixels or cx + w > img_w-epsilon_pixels or cy + h > img_h-epsilon_pixels:
        return corrected_labels
    
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    transform_tag = xml_path.replace('.xml','.jpg')
    # If the other image was labeled take transform from that one
    if labeled is not None and labeled not in transform_tag:
        transform_tag.replace('visible', labeled).replace('lwir', labeled)
    
    transform_matrix = load_transform(transform_tag)
    points_array = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    points_array = points_array.reshape(-1, 1, 2)
    transformed_points = cv2.transform(points_array, transform_matrix)
    (x1, y1), (x2, y2) = transformed_points.reshape(-1, 2)

    # Correct label to new image position and size
    # New label center should be alwais in the image
    new_x1 = min(crop_w-epsilon_pixels, max(epsilon_pixels, x1 - crop_x))
    new_y1 = min(crop_h-epsilon_pixels, max(epsilon_pixels, y1 - crop_y))
    new_x2 = min(crop_w-epsilon_pixels, max(epsilon_pixels, x2 - crop_x))
    new_y2 = min(crop_h-epsilon_pixels, max(epsilon_pixels, y2 - crop_y))
    
    if new_x1 < new_x2 and new_y1 < new_y2:
        for dx, dy in product([0, crop_w], [0, crop_h]):
            mosaic_new_x1 = min(img_w, max(dx, new_x1 + dx))
            mosaic_new_y1 = min(img_h, max(dy, new_y1 + dy))
            mosaic_new_x2 = max(dx, min(img_w, new_x2 + dx))
            mosaic_new_y2 = max(dy, min(img_h, new_y2 + dy))
            if mosaic_new_x1 < mosaic_new_x2 and mosaic_new_y1 < mosaic_new_y2:
                mosaic_new_w = mosaic_new_x2 - mosaic_new_x1
                mosaic_new_h = mosaic_new_y2 - mosaic_new_y1
                mosaic_new_cx = mosaic_new_x1 + mosaic_new_w/2
                mosaic_new_cy = mosaic_new_y1 + mosaic_new_h/2
                corrected_labels.append([obj_id, mosaic_new_cx, mosaic_new_cy, mosaic_new_w, mosaic_new_h])
        
    return corrected_labels

def applyUndistort(points, calibration_matrix = camera_matrix_est, dist_coeffs = dist_coeffs_est):
    original_coord_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(original_coord_np, calibration_matrix, dist_coeffs).reshape(-1, 1, 2)
    undistorted_points = cv2.convertPointsToHomogeneous(undistorted_points).reshape(-1, 3)
    undistorted_points = (calibration_matrix @ undistorted_points.T).T
    undistorted_points = undistorted_points[:, :2]  # Ignorar la tercera coordenada
    return undistorted_points

