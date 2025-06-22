
#!/usr/bin/env python3
# encoding: utf-8
"""
    Creates different approachs of mixing RGB with Thermal images:
    Wavelet Transform:
    MDMR:
"""

import os
import time

import cv2 as cv
import numpy as np
import pywt

from sklearn.decomposition import PCA
from curvelets.numpy import SimpleUDCT

# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')


from utils import log, bcolors
from Dataset.decorators import time_execution_measure, save_image_if_path, save_npmat_if_path
from Dataset.fusion_methods.normalization import normalize

@save_npmat_if_path
def combine_hsvt_wavelet(visible_image, thermal_image):
    h, s, v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = cv.normalize(thermal_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    
    # Aplicar la Transformada Discreta Wavelet a los canales de intensidad (V y térmico)
    coeffs_v = pywt.dwt2(v, 'haar')
    coeffs_th = pywt.dwt2(th_channel, 'haar')
    
    # Fuse (average of coeffs)
    cA_fused = (coeffs_v[0] + coeffs_th[0]) / 2
    cH_fused = (coeffs_v[1][0] + coeffs_th[1][0]) / 2  # Horizontal average
    cV_fused = (coeffs_v[1][1] + coeffs_th[1][1]) / 2  # Vertical average
    cD_fused = (coeffs_v[1][2] + coeffs_th[1][2]) / 2  # Diagonal average
    
    # Reconstruct with inverse transform
    fused_intensity = pywt.idwt2((cA_fused, (cH_fused, cV_fused, cD_fused)), 'haar')
    
    # Re-normalize to [0-255)]
    fused_intensity = np.clip(fused_intensity, 0, 255).astype(np.uint8)
    
    hsvt_image = cv.merge([h, s, fused_intensity])
    hsvt_image = cv.cvtColor(hsvt_image, cv.COLOR_HSV2BGR)
    
    return hsvt_image


@save_npmat_if_path
def combine_rgb_wavelet(visible_image, thermal_image):
    rgbt = np.dstack((visible_image, thermal_image))

    # Wavelet decoposition in approximation sub-bands (low freq.) and detail sub-bands (high freq.).
    # Apply Discret Wavelet Transform to each channel
    coeffs = [pywt.dwt2(rgbt[:,:,i], 'haar') for i in range(4)]
    cAprox = [c[0] for c in coeffs]
    cDetail = [c[1] for c in coeffs]

    # Fuse coefficients
    # Detail sub-bads capture mostly: bordes, texturas y cambios locales
    cDetail_fused = []
    for i in range(3):  # Fuse for 3 output channels
        # Averaged with thermal channel coeffs? Max?
        # Average tends to lose texture and borders, abs max better preserve these features
        cH = np.where(np.abs(cDetail[i][0]) > np.abs(cDetail[3][0]), cDetail[i][0], cDetail[3][0])
        cV = np.where(np.abs(cDetail[i][1]) > np.abs(cDetail[3][1]), cDetail[i][1], cDetail[3][1])
        cD = np.where(np.abs(cDetail[i][2]) > np.abs(cDetail[3][2]), cDetail[i][2], cDetail[3][2])
        cDetail_fused.append((cH, cV, cD))

    ## Aproximation sub-bands contain most of the information. Thats why a better fusion is applied
    # Las bandas de aproximación contienen la mayor parte de la información estructural y espectral 
    # relevante de la imagen.
    
    # For aproximation coef if pixel in thermal image is high, more weight to thermal coeff
    weights = thermal_image / (thermal_image.max() + 1e-8)
    weights_resized = cv.resize(weights, (cAprox[3].shape[1], cAprox[3].shape[0]), interpolation=cv.INTER_LINEAR)
    cA_fused = []
    for i in range(3):
        cA_fused.append(weights_resized * cAprox[3] + (1 - weights_resized) * cAprox[i])

    fused_image = []
    for i in range(3):
        coeffs = (cA_fused[i], cDetail_fused[i])
        fused_channel = pywt.idwt2(coeffs, 'haar')
        fused_channel = fused_channel[:visible_image.shape[0], :visible_image.shape[1]] # Ensure original shape
        fused_image.append(fused_channel)
        
    fused_image = np.stack(fused_image, axis=-1) 
    
    # Ensure range 0-255 and uint8 encoding
    fused_image = normalize(fused_image)
    return fused_image


@save_npmat_if_path
def combine_hsv_curvelet(visible_image, thermal_image):
    h, s, v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    
    # Apply curvelet to V and thermal channel
    winthresh = 1e-5
    curvelet_transform = SimpleUDCT(shape=thermal_image.shape, nscales=4, nbands_per_direction=4, alpha=0.3, winthresh=winthresh)
    c_v = curvelet_transform.forward(v)
    c_th = curvelet_transform.forward(thermal_image)

    # print("c_v:", c_v)
    # print("c_th:", c_th)
    
    # Average coeffs
    c_fused = []
    for i in range(len(c_v)):  # Iter each level
        level_fused = []
        for j in range(len(c_v[i])):  # Iter each coeff in level
            # Extract real part (magnitud)
            coeff_v = np.abs(c_v[i][j])
            coeff_th = np.abs(c_th[i][j])

            # Fuse by averaging
            fused_coeff = (coeff_v + coeff_th) / 2

            level_fused.append(fused_coeff)
        c_fused.append(level_fused)

        
    # Reconstruct image
    fused_intensity = curvelet_transform.backward(c_fused)
    
    fused_intensity = np.clip(fused_intensity, 0, 255).astype(np.uint8)
    hsvt_image = cv.merge([h, s, fused_intensity])
    hsvt_image = cv.cvtColor(hsvt_image, cv.COLOR_HSV2BGR)
    
    return hsvt_image

@save_npmat_if_path
def combine_rgb_curvelet(visible_image, thermal_image):
    rgbt = np.dstack((visible_image, thermal_image))
    
    # Curvelet transform for each channel
    winthresh = 1e-5
    curvelet_transform = SimpleUDCT(
        shape=thermal_image.shape, nscales=4, nbands_per_direction=4, alpha=0.3, winthresh=winthresh
    )
    # Image is decomposed in nscales levels
    coeffs = [curvelet_transform.forward(rgbt[:, :, i]) for i in range(4)]
    
    # Fuse coefficients
    fused_coeffs = []
    for i in range(3):  # 3 output channels
        c_fused = []
        for j in range(len(coeffs[i])):  # Iterate levels (nscales)
            scale_fused = []
            for k in range(len(coeffs[i][j])):  # Iterate coefficients in each level
                coeff_v = np.array(coeffs[i][j][k])  # Ensure numpy array
                coeff_th = np.array(coeffs[3][j][k])  # Thermal channel as numpy array
                
                if j == 0:
                    # Aproximation (low freq.)
                    fused_coeff = (coeff_v + coeff_th) / 2
                else:
                    # Detail (high freq.)
                    mask = np.abs(coeff_v) >= np.abs(coeff_th)
                    fused_coeff = np.where(mask, coeff_v, coeff_th)

                scale_fused.append(fused_coeff)
            c_fused.append(scale_fused)
        fused_coeffs.append(c_fused)
    
    # Reconstruct fused channels
    fused_channels = [curvelet_transform.backward(coeff) for coeff in fused_coeffs]
    
    # Stack channels and normalize
    fused_image = np.stack(fused_channels, axis=-1)
    fused_image = normalize(fused_image)
    #fused_image = cv.cvtColor(fused_image, cv.COLOR_RGB2BGR)
    
    return fused_image 




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    lwir_image_path = '/home/arvc/eeha/kaist-cvpr15/images/set00/V000/lwir/I01689.jpg'
    visible_image_path = lwir_image_path.replace('/lwir/', '/visible/')
    
    visible_image = cv.imread(visible_image_path)
    lwir_image = cv.imread(lwir_image_path, cv.IMREAD_GRAYSCALE)

    decorated_function = time_execution_measure(combine_hsvt_wavelet)
    hsvt_wavelet_image, hsvt_wavelet_time_execution = decorated_function(visible_image, lwir_image)
    
    decorated_function = time_execution_measure(combine_rgb_wavelet)
    rgb_wavelet_image, rgb_wavelet_time_execution = decorated_function(visible_image, lwir_image)
    
    decorated_function = time_execution_measure(combine_hsv_curvelet)
    hsv_curvelet_image, hsv_curvelet_time_execution = decorated_function(visible_image, lwir_image)
    
    decorated_function = time_execution_measure(combine_rgb_curvelet)
    rgb_curvelet_image, rgb_curvelet_time_execution = decorated_function(visible_image, lwir_image)
    
    data = [
        [visible_image, f'Original image (Visible)'],
        [hsvt_wavelet_image, f'HSV+Wavelet ({hsvt_wavelet_time_execution:.2f} s)'],
        [hsv_curvelet_image, f'HSV+Curvelet ({hsv_curvelet_time_execution:.2f} s)'],
        [lwir_image, f'Original image (LWIR)'],
        [rgb_wavelet_image, f'RGB+Wavelet ({rgb_wavelet_time_execution:.2f} s)'],
        [rgb_curvelet_image, f'RGB+Curvelet ({rgb_curvelet_time_execution:.2f} s)']
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    for ax, (img, title) in zip(axes.flatten(), data):
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    
    # plt.tight_layout()
    plt.show()
