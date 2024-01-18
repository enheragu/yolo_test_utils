#!/usr/bin/env python3
# encoding: utf-8
"""
    Creates different approachs of mixing RGB with Thermal images
    PCA -> Principal component analysis (based on covariation between channels)
    FA -> Factorial analysis (based on correlation between channels)
"""

import numpy as np
import cv2 

from config_utils import log

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

"""
    Common matrix analsisi for PCA and FA for different use cases
"""
def MatrixAnalisis(data_vector, mat, img_shape, components, standarice = True, path = None):
    if standarice:
        data_vector_std = (data_vector - data_vector.mean()) / data_vector.std() # standarize || axis = 0 -> Is it not needed?
    else:
        data_vector_std = data_vector
    mat = np.cov(data_vector_std, ddof = 1, rowvar = False)
    # eigenvalues, eigenvectors = np.linalg.eig(mat)
    eigenvalues, eigenvectors = np.linalg.eigh(mat) # -> faster but only can be used for symmetric matrix
    # eith returns eigenvalues in ascendin order already
    
    signs = np.sign(eigenvectors[0, :]) # extract sign of each first component of each column
    # Multiply matriz with them to make all first component positive
    eigenvectors = eigenvectors * signs

    # order_of_importance = np.argsort(eigenvalues)[::-1] # Order eigenvalues high to low
    # sorted_eigenvalues = eigenvalues[order_of_importance]
    # sorted_eigenvectors = eigenvectors[:,order_of_importance]
    sorted_eigenvalues = np.flip(eigenvalues)
    sorted_eigenvectors = np.flip(eigenvectors, axis = 1) # eigenvectors are in columns of the matrix, flip in that axis

    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    k = components # select the number of principal components
    image = np.matmul(data_vector_std, sorted_eigenvectors[:,:k]) # transform the original data
    
    # Re-escale intensity
    if standarice:
        image = image * data_vector.std() + data_vector.mean() # axis = 0 -> is it not needed?
    

    ## Store eigenvalue and vectors to file to later study
    # autov_path = path.split("/")[:-1]
    # autov_path = "/".join(autov_path) + "/00_eigenvalue_vector.yaml"

    # import yaml
    # from yaml.loader import SafeLoader
    # import os

    # data = {}
    # if os.path.exists(autov_path) and os.path.isfile(autov_path):
    #     with open(autov_path, 'r') as file:
    #         data = yaml.safe_load(file)
    #         data = {} if data is None else data
    
    # data[path] = {"img": str(path), "eigenvectors": (eigenvectors.transpose().tolist()), "eigenvalues": (eigenvalues.tolist())}
    # with open(autov_path, 'w') as file:
    #     yaml.safe_dump(data, file)

    # total_explained_variance = sum(explained_variance[:k])
    # log(f"[RGBThermalMix::combine_pca] Explained variance for {path} with {k} principal components is: {total_explained_variance}")
    # print(f"{sorted_eigenvectors[:,:k] = }")

    image_vector = []
    for i in range(components):
        image_vector.append(image.transpose()[i].reshape(img_shape))
        
    image = cv2.merge(image_vector)
    
    # # if test:
    # condition_01 = sorted_eigenvalues[0] / sorted_eigenvalues[1]
    # condition_12 = sorted_eigenvalues[1] / sorted_eigenvalues[2]
    # condition_23 = sorted_eigenvalues[2] / sorted_eigenvalues[3]
    # w, h = draw_text(image, f"eigenvalues[0]={np.around(eigenvalues[0],6)}", pos=(10, 10))
    # w, h = draw_text(image, f"eigenvalues[1]={np.around(eigenvalues[1],6)}", pos=(10, 20 + h))
    # w, h = draw_text(image, f"eigenvalues[2]={np.around(eigenvalues[2],6)}", pos=(10, 40 + h))
    # w, h = draw_text(image, f"eigenvalues[3]={np.around(eigenvalues[3],6)}", pos=(10, 60 + h))
    # # Eigenvectors are in the columns, transpose it first
    # w, h = draw_text(image, f"eigenvectors[0]={list(np.around(np.array(eigenvectors.transpose()[0]),6))}", pos=(10, 80 + h))
    # w, h = draw_text(image, f"eigenvectors[1]={list(np.around(np.array(eigenvectors.transpose()[1]),6))}", pos=(10, 100 + h))
    # w, h = draw_text(image, f"eigenvectors[2]={list(np.around(np.array(eigenvectors.transpose()[2]),6))}", pos=(10, 120 + h))
    # w, h = draw_text(image, f"eigenvectors[3]={list(np.around(np.array(eigenvectors.transpose()[3]),6))}", pos=(10, 140 + h))
    # w, h = draw_text(image, f"{explained_variance[0] = }", pos=(10, 160 + h))
    # w, h = draw_text(image, f"{explained_variance[1] = }", pos=(10, 180 + h))
    # w, h = draw_text(image, f"{explained_variance[2] = }", pos=(10, 200 + h))
    # w, h = draw_text(image, f"{explained_variance[3] = }", pos=(10, 220 + h))
    
    return image

def combine_rgbt_pca_toXch(visible_image, thermal_image, path, output_channels = 3):

    # NOISE FILTERING WITH BETTER EDGE PRESERVATION
    visible_imgage_filered = cv2.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = cv2.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv2.split(visible_imgage_filered)
    th = cv2.cvtColor(thermal_image_filtered, cv2.COLOR_BGR2GRAY)
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()   
    cov_mat = np.cov(data_vector, ddof = 1, rowvar = False) 
    
    image = MatrixAnalisis(data_vector, cov_mat, img_shape, components = output_channels, standarice = True, path = path)

    return image


def combine_rgbt_fa_toXch(visible_image, thermal_image, path, output_channels = 3):
    
    # NOISE FILTERING WITH BETTER EDGE PRESERVATION
    visible_imgage_filered = cv2.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = cv2.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv2.split(visible_imgage_filered)
    th = cv2.cvtColor(thermal_image_filtered, cv2.COLOR_BGR2GRAY)
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()

    # cov_mat = np.corrcoef(data_vector_std) # -> Pearson correlation. Careful!!
    import scipy.stats
    cov_mat, p_matrix = scipy.stats.spearmanr(data_vector, axis=0) # -> Spearman. axis whether columns (0) or rows (1) represent the features

    image = MatrixAnalisis(data_vector, cov_mat, img_shape, components = output_channels, standarice = False, path = path)

    return image

def combine_rgbt_pca_to3ch(visible_image, thermal_image, path):
    image = combine_rgbt_pca_toXch(visible_image, thermal_image, path, 3)
    cv2.imwrite(path, image)
    return image

def combine_rgbt_pca_to2ch(visible_image, thermal_image, path):
    image = combine_rgbt_pca_toXch(visible_image, thermal_image, path, 2)
    np.save(path.replace('.png',''), image)
    return image    

def combine_rgbt_pca_to1ch(visible_image, thermal_image, path):
    image = combine_rgbt_pca_toXch(visible_image, thermal_image, path, 1)
    np.save(path.replace('.png',''), image)
    return image    

def combine_rgbt_fa_to3ch(visible_image, thermal_image, path):
    return combine_rgbt_fa_toXch(visible_image, thermal_image, path, 3)
    cv2.imwrite(path, image)
    return image    

def combine_rgbt_fa_to2ch(visible_image, thermal_image, path):
    image = combine_rgbt_fa_toXch(visible_image, thermal_image, path, 2)
    np.save(path.replace('.png',''), image)
    return image    

def combine_rgbt_fa_to1ch(visible_image, thermal_image, path):
    image = combine_rgbt_fa_toXch(visible_image, thermal_image, path, 1)
    np.save(path.replace('.png',''), image)
    return image    


def combine_hsvt_pca_to3ch(visible_image, thermal_image, path):
    # NOISE FILTERING WITH BETTER EDGE PRESERVATION
    visible_imgage_filered = cv2.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = cv2.bilateralFilter(thermal_image, 9, 50, 50) 

    h,s,v = cv2.split(cv2.cvtColor(visible_image, cv2.COLOR_BGR2HSV))
    th = cv2.cvtColor(thermal_image_filtered, cv2.COLOR_BGR2GRAY)
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [h,s,v,th]]).transpose()   
    cov_mat = np.cov(data_vector, ddof = 1, rowvar = False) 
    
    image = MatrixAnalisis(data_vector, cov_mat, img_shape, components = 3, standarice = True, path = path)

    return image


options = {'pca_rgbt_1ch' : {'merge': combine_rgbt_pca_to1ch, 'extension': '.npy' },
           'pca_rgbt_2ch' : {'merge': combine_rgbt_pca_to2ch, 'extension': '.npy' },
           'pca_rgbt_3ch' : {'merge': combine_rgbt_pca_to3ch, 'extension': '.png' },
        #    'pca_hsvt_3ch' : {'merge': combine_hsvt_pca_to3ch, 'extension': '.png' }, # -> Result is really bad, makes no sense to look for covariance in that format
           'fa_rgbt_3ch' : {'merge': combine_rgbt_fa_to3ch, 'extension': '.png' },
           'fa_rgbt_2ch' : {'merge': combine_rgbt_fa_to2ch, 'extension': '.npy' },
           'fa_rgbt_1ch' : {'merge': combine_rgbt_fa_to1ch, 'extension': '.npy' }
          }
