#!/usr/bin/env python3
# encoding: utf-8
"""
    Creates different approachs of mixing RGB with Thermal images
    PCA -> Principal component analysis (based on covariation between channels)
    FA -> Factorial analysis (based on correlation between channels)
"""

import numpy as np
import cv2 as cv

from utils import log, bcolors
from Dataset.decorators import time_execution_measure

def draw_text(img, text,
          font=cv.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

"""
    Common matrix analsisi for PCA and FA for different use cases
"""
def MatrixAnalisis(data_vector, mat, img_shape, components, standarice = True):
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
        
    image = cv.merge(image_vector)
    
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


def combine_rgbt_pca_toXch(visible_image, thermal_image, output_channels = 3):

    # NOISE FILTERING WITH BETTER EDGE PRESERVATION
    visible_imgage_filered = cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = cv.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv.split(visible_imgage_filered)
    th = thermal_image_filtered
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()   
    cov_mat = np.cov(data_vector, ddof = 1, rowvar = False) 
    
    image = MatrixAnalisis(data_vector, cov_mat, img_shape, components = output_channels, standarice = True)
    image = image.astype(np.uint8) # Recast to image format type after all operations
    
    return image


def combine_rgbt_fa_toXch(visible_image, thermal_image, output_channels = 3):
    
    # NOISE FILTERING WITH BETTER EDGE PRESERVATION
    visible_imgage_filered = cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = cv.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv.split(visible_imgage_filered)
    th = thermal_image_filtered
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()

    # cov_mat = np.corrcoef(data_vector_std) # -> Pearson correlation. Careful!!
    import scipy.stats
    cov_mat, p_matrix = scipy.stats.spearmanr(data_vector, axis=0) # -> Spearman. axis whether columns (0) or rows (1) represent the features

    image = MatrixAnalisis(data_vector, cov_mat, img_shape, components = output_channels, standarice = False)
    image = image.astype(np.uint8) # Recast to image format type after all operations

    return image

@time_execution_measure
@save_npmat_if_path
def combine_rgbt_pca_to3ch(visible_image, thermal_image):
    image = combine_rgbt_pca_toXch(visible_image, thermal_image, 3)
    return image

# @time_execution_measure
# @save_npmat_if_path
# def combine_rgbt_pca_to2ch(visible_image, thermal_image):
#     image = combine_rgbt_pca_toXch(visible_image, thermal_image, 2)
#     return image    

# @time_execution_measure
# @save_npmat_if_path
# def combine_rgbt_pca_to1ch(visible_image, thermal_image):
#     image = combine_rgbt_pca_toXch(visible_image, thermal_image, 1)
#     return image    

@time_execution_measure
@save_npmat_if_path
def combine_rgbt_fa_to3ch(visible_image, thermal_image):
    image = combine_rgbt_fa_toXch(visible_image, thermal_image, 3)
    return image    

# def combine_rgbt_fa_to2ch(visible_image, thermal_image):
#     image = combine_rgbt_fa_toXch(visible_image, thermal_image, 2)
#     # np.savez_compressed(path.replace('.png',''), image = image)
#     np.save(path.replace('.png',''), image)
#     return image    

# def combine_rgbt_fa_to1ch(visible_image, thermal_image):
#     image = combine_rgbt_fa_toXch(visible_image, thermal_image, 1)
#     # np.savez_compressed(path.replace('.png',''), image = image)
#     np.save(path.replace('.png',''), image)
#     return image    

@time_execution_measure
@save_npmat_if_path
def combine_hsvt_pca_to3ch(visible_image, thermal_image):
    # NOISE FILTERING WITH BETTER EDGE PRESERVATION
    visible_imgage_filered = cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = cv.bilateralFilter(thermal_image, 9, 50, 50) 

    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th = thermal_image_filtered
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [h,s,v,th]]).transpose()   
    cov_mat = np.cov(data_vector, ddof = 1, rowvar = False) 
    
    image = MatrixAnalisis(data_vector, cov_mat, img_shape, components = 3, standarice = True)

    return image


"""
    Gets PCA eigenvectors and eigenvalues for given dataset
"""
pca_output_path = '~/.cache/eeha_yolo_test/pca_sorted_eigenvectors.npy'
fa_output_path = '~/.cache/eeha_yolo_test/fa_sorted_eigenvectors.npy'
pca_eigen_vectors = None        # Avoid multiple load of data. Load once and reuse
fa_eigen_vectors = None

def preprocess_rgbt_pca_full(option, dataset_format):
    log(f"Compute general PCA eigenvectors and values to later compress images.")
    pass

"""
    Gets FA eigenvectors and eigenvalues for given dataset
"""
def preprocess_rgbt_fa_full(option, dataset_format):
    log(f"Compute general FA eigenvectors and values to later compress images.")
    pass

@time_execution_measure
def combine_rgbt_pca_full(visible_image, thermal_image):
    global pca_eigen_vectors
    visible_imgage_filered = cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = cv.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv.split(visible_imgage_filered)
    th = thermal_image_filtered
    img_shape = th.shape

    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()   
    data_vector_std = (data_vector - data_vector.mean()) / data_vector.std() # standarize 
    
    if not pca_eigen_vectors:
        pca_eigen_vectors = np.load(pca_output_path)
    image = np.matmul(data_vector_std, pca_eigen_vectors[:,:4])

    np.save(path.replace('.png',''), image)
    return image

@time_execution_measure
def combine_rgbt_fa_full(visible_image, thermal_image):
    global fa_eigen_vectors
    visible_imgage_filered = cv.bilateralFilter(visible_image, 9, 50, 50) 
    thermal_image_filtered = cv.bilateralFilter(thermal_image, 9, 50, 50) 

    b,g,r = cv.split(visible_imgage_filered)
    th = cthermal_image_filtered
    img_shape = th.shape

    if not fa_eigen_vectors:
        fa_eigen_vectors = np.load(fa_output_path)
    data_vector = np.array([f.flatten() for f in [b,g,r,th]]).transpose()   
    image = np.matmul(data_vector, fa_eigen_vectors[:,:4])

    np.save(path.replace('.png',''), image)
    return image