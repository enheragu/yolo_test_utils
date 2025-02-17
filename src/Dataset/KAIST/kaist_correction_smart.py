
import os
import cv2
import numpy as np
from itertools import product
import pickle
from pathlib import Path

from utils import log, bcolors
from Dataset.constants import kaist_path, images_folder_name, labels_folder_name

image_shape = (640,512)

optical_flow_cache = {}
def load_optical_flow(rgb_file_path, optical_flow_data_path = '/home/arvc/eeha/multiespectral_correction/data'):
    global optical_flow_cache

    set_name = Path(rgb_file_path).parts[1]
    sequence_name = Path(rgb_file_path).parts[2]
    if f"{set_name}/{sequence_name}" not in optical_flow_cache:
        optical_flow_file = f'{optical_flow_data_path}/optical_flow_{set_name}_{sequence_name}.pkl'

        with open(optical_flow_file, 'rb') as file:
            data = pickle.load(file)
        optical_flow_cache[f"{set_name}/{sequence_name}"] = data
    optical_flow_list = optical_flow_cache[f"{set_name}/{sequence_name}"]
    flow_data = next(item for item in optical_flow_list if item['visible'] == rgb_file_path)
    return flow_data

# Default average_fraction extracted from:
#  https://github.com/enheragu/multiespectral_correction:src/02_computeMultiespectralDistortionFactor.py
def scaleAffineTransform(T, average_fraction = 5):
    T = np.array(T)
    A = T[:2, :2]
    t = T[:2, 2]
    
    I = np.eye(2)
    A_scaled = I + (A - I) * average_fraction
    t_scaled = t * average_fraction
    
    T_scaled = np.eye(3)[:2, :]  # Create a 2x3 matrix
    T_scaled[:2, :2] = A_scaled
    T_scaled[:2, 2] = t_scaled
    
    return T_scaled

def invertAffineTransform(transform):
    transform_homogeneous = np.vstack([transform, [0, 0, 1]])
    inverted_homogeneous = np.linalg.inv(transform_homogeneous)
    inverted_transform = inverted_homogeneous[:2, :]
    return inverted_transform

# Keep at least 65% of the original image, max_crop to 35% 
# Default Maximum data extracted from:
#  https://github.com/enheragu/multiespectral_correction:src/02_computeMultiespectralDistortionFactor.py
import numpy as np
def getImageCrop(translation_x, translation_y, image_shape=image_shape, 
                 max_crop=0.35,
                 translation_limits_x = [-4.818207761938845, 12.099579119016507],
                 translation_limits_y = [-2.318536707291676, 6.618311282898231]
                ):
    
    w, h = image_shape
    img_center_x = w / 2
    img_center_y = h / 2
    
    translation_x = np.clip(translation_x,translation_limits_x[0],translation_limits_x[1])
    translation_y = np.clip(translation_y,translation_limits_y[0],translation_limits_y[1])
    translation_x_norm = (translation_x - translation_limits_x[0]) / (translation_limits_x[1] - translation_limits_x[0]) # normalization 0-1
    translation_y_norm = (translation_y - translation_limits_y[0]) / (translation_limits_y[1] - translation_limits_y[0]) # normalization 0-1

    # crop_factor_ = max_crop * [-1,1] translation
    crop_factor_w = max_crop * (2 * translation_x_norm - 1) 
    crop_factor_h = max_crop * (2 * translation_y_norm - 1) 

    crop_w = int(w * (1 - abs(crop_factor_w)))
    crop_h = int(h * (1 - abs(crop_factor_h)))

    new_center_limit = [[0+crop_w/2,0+crop_h/2], [w-crop_w/2,h-crop_h/2]]
    max_displacement_x = (new_center_limit[1][0]-new_center_limit[0][0])/2
    max_displacement_y = (new_center_limit[1][0]-new_center_limit[0][0])/2

    crop_center_x = img_center_x + max_displacement_x * (2 * translation_x_norm - 1)
    crop_center_y = img_center_y + max_displacement_y * (2 * translation_y_norm - 1)
    
    crop_center_x = np.clip(crop_center_x, new_center_limit[0][0], new_center_limit[1][0])
    crop_center_y = np.clip(crop_center_y, new_center_limit[0][1], new_center_limit[1][1])
    
    # New coordinates of crop vertex :D
    crop_x = int(crop_center_x - crop_w / 2)
    crop_y = int(crop_center_y - crop_h / 2)

    print(f"{crop_x = }, {crop_y = }, {crop_w = }, {crop_h = }; {translation_x = }; {translation_y = }; {crop_center_x = }; {crop_center_y = }")

    return crop_x, crop_y, crop_w, crop_h



def fillBorders(image, x, y, w, h):
    visible_part = image[y:y+h, x:x+w]
    filled_image = np.ones_like(image) * 255
    filled_image[y:y+h, x:x+w] = visible_part
    
    return filled_image
    # Determinar el número de canales
    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1
    
    if y > 0:
        if channels == 1:
            filled_image[:y, x:x+w] = np.tile(visible_part[0:1, :], (y, 1))
        else:
            filled_image[:y, x:x+w] = np.tile(visible_part[0:1, :, :], (y, 1, 1))
    
    if y + h < image.shape[0]:
        if channels == 1:
            filled_image[y+h:, x:x+w] = np.tile(visible_part[-1:, :], (image.shape[0] - (y+h), 1))
        else:
            filled_image[y+h:, x:x+w] = np.tile(visible_part[-1:, :, :], (image.shape[0] - (y+h), 1, 1))
    
    if x > 0:
        filled_image[:, :x] = np.tile(filled_image[:, x:x+1], (1, x, 1) if channels > 1 else (1, x))
    
    if x + w < image.shape[1]:
        filled_image[:, x+w:] = np.tile(filled_image[:, x+w-1:x+w], (1, image.shape[1] - (x+w), 1) if channels > 1 else (1, image.shape[1] - (x+w)))
    
    return filled_image


def generateCorrectedImage(image, data_type, image_path):
    image_data_tag = image_path.replace('lwir','visible').replace(os.path.join(kaist_path,images_folder_name),'')
        
    corrected_image = image.copy()

    flow_data = load_optical_flow(image_data_tag)
    visible_flow = flow_data['oflow_visible']

    if visible_flow is not None:
        if 'lwir' in data_type:
            visible2visible_transform = np.array(visible_flow['transformation_matrix'])
            visible2lwir_transform = scaleAffineTransform(invertAffineTransform(visible2visible_transform))
            lwir2visible_transform = invertAffineTransform(visible2lwir_transform)
            rows, cols = image.shape[:2]
            corrected_image = cv2.warpAffine(image, lwir2visible_transform, (cols, rows))

        x, y, w, h = getImageCrop(visible_flow['translation_x'], visible_flow['translation_y'])
        corrected_image = fillBorders(corrected_image,x,y,w,h)
        
    return corrected_image

def generateCorrectedLabels(label, xml_path, image_shape = image_shape, labeled_visible=False):
    img_w, img_h = image_shape
    corrected_labels = []
    obj_id, obj_x, obj_y, obj_w, obj_h = label

    image_data_tag = xml_path.replace('lwir','visible').replace('.xml', '.jpg')
    print(image_data_tag)
    flow_data = load_optical_flow(image_data_tag)
    visible_flow = flow_data['oflow_visible']

    if visible_flow is not None:
        x, y, w, h = getImageCrop(visible_flow['translation_x'], visible_flow['translation_y'])

        # Adjust input label to cropped image
        new_x = max(0, min(w, obj_x - x))
        new_y = max(0, min(h, obj_y - y))
        new_w = min(w - new_x, obj_w)
        new_h = min(h - new_y, obj_h)

        if new_w > 0 and new_h > 0:
            corrected_labels.append([obj_id, new_x + x, new_y + y, new_w, new_h])

        # Verify if labels get into filled areas
        if obj_x < x:
            left_label = [obj_id, obj_x + img_w, obj_y, min(obj_w, x - obj_x), obj_h]
            corrected_labels.append(left_label)
        
        if obj_x + obj_w > x + w:
            right_label = [obj_id, obj_x - w, obj_y, min(obj_w, obj_x + obj_w - (x + w)), obj_h]
            corrected_labels.append(right_label)
        
        if obj_y < y:
            top_label = [obj_id, obj_x, obj_y + img_h, obj_w, min(obj_h, y - obj_y)]
            corrected_labels.append(top_label)
        
        if obj_y + obj_h > y + h:
            bottom_label = [obj_id, obj_x, obj_y - h, obj_w, min(obj_h, obj_y + obj_h - (y + h))]
            corrected_labels.append(bottom_label)
    else:
        corrected_labels.append(label)

    return corrected_labels

