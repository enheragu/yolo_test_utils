#!/usr/bin/env python3
# encoding: utf-8
"""
    Fusino method that adjusts the thermal contribution based on the structural similarity index (SSIM)
"""


import os
import numpy as np
import cv2 as cv

from skimage.metrics import structural_similarity as ssim

from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import filters
from scipy.ndimage import gaussian_filter
from scipy.special import expit 

from utils import log, bcolors
from Dataset.decorators import time_execution_measure, save_image_if_path, save_npmat_if_path

@save_image_if_path
def combine_rgbt_ssim(visible, thermal, win_size=11, block_size=32):
    # Extracts SSMI (Structural Similarity Index) for each channel of the visible image 
    # with respect to the thermal image
    ssim_r, ssim_r_map = ssim(visible[..., 0], thermal, full=True, win_size=win_size)
    ssim_g, ssim_g_map = ssim(visible[..., 1], thermal, full=True, win_size=win_size)
    ssim_b, ssim_b_map = ssim(visible[..., 2], thermal, full=True, win_size=win_size)

    fused = np.empty_like(visible, dtype=float)
    h, w = visible.shape[:2]

    def ssim_map_smooth_norm(ssim_map):
        ssim_map_smooth = gaussian_filter(ssim_map, sigma=2)
        ssim_map_norm = (ssim_map_smooth - ssim_map_smooth.min()) / (ssim_map_smooth.max() - ssim_map_smooth.min() + 1e-8)
        return ssim_map_norm

    ssim_r_map = ssim_map_smooth_norm(ssim_r_map)
    ssim_g_map = ssim_map_smooth_norm(ssim_r_map)
    ssim_b_map = ssim_map_smooth_norm(ssim_r_map)

    # Adaptative fusion based on SSIM
    # Fuse for each pixel based on the SSIM map value
    fused[..., 0] = (1 - ssim_r_map) * visible[..., 0] + ssim_r_map * thermal
    fused[..., 1] = (1 - ssim_g_map) * visible[..., 1] + ssim_g_map * thermal
    fused[..., 2] = (1 - ssim_b_map) * visible[..., 2] + ssim_b_map * thermal

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused

@save_image_if_path
def combine_rgbt_sobel_weighted(visible, thermal, alpha=0.5):
    grad_thermal = filters.sobel(thermal)
    grad_thermal_norm = (grad_thermal - grad_thermal.min()) / (grad_thermal.ptp() + 1e-8)  # Normalizar a [0,1]

    # Fusión ponderada por el gradiente
    fused = np.empty_like(visible, dtype=float)
    for c in range(3):
        fused[..., c] = (1 - alpha * grad_thermal_norm) * visible[..., c] + (alpha * grad_thermal_norm) * thermal

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused


# ReviewED FROM https://ieeexplore.ieee.org/document/10095874
@save_image_if_path
def combine_rgbt_superpixel(visible, thermal):
    n_segments=200
    sigma=2.0

    # 1. Superpixel segmentation based on averaged RGB image 
    gray_visible = np.mean(visible, axis=2)
    base_for_slic = (gray_visible.astype(float) + thermal.astype(float)) / 2
    
    r,g,b = cv.split(visible)
    base_for_slic = cv.merge([r,g,b,thermal])
    # SLIC (Simple Linear Iterative Clustering) segmentation (clusters close pixels based on similarity)
    segments = slic(base_for_slic, n_segments=n_segments, compactness=10) 
    mask = np.zeros(visible.shape[:2], dtype=float)

    # 2. Compute deviation per superpixel for each image
    grades_visible = np.zeros_like(mask)
    grades_thermal = np.zeros_like(mask)
    for seg_val in np.unique(segments):
        mask_sp = segments == seg_val
        # For visible images combines channels
        grades_visible[mask_sp] = np.mean([np.std(visible[..., c][mask_sp]) for c in range(visible.shape[2])])
        grades_thermal[mask_sp] = np.std(thermal[mask_sp])

    # 3. Weights map: difference from previous; then norm and smoothened
    diff = grades_visible - grades_thermal
    mask = expit(diff)  # Sigmoid
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = gaussian_filter(mask, sigma=sigma)

    # 4. Fused image for each RGB channel with thermal contribution
    fused = np.empty_like(visible, dtype=float)
    for c in range(visible.shape[2]):
        fused[..., c] = mask * visible[..., c] + (1 - mask) * thermal

    fused = np.clip(fused, 0, 255).astype(visible.dtype)
    return fused