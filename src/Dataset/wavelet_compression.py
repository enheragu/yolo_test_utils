#!/usr/bin/env python3
# encoding: utf-8
"""
    Creates different approachs of mixing RGB with Thermal images with wavelet trnasform
"""

import pywt
import numpy as np

from Dataset.decorators import time_execution_measure, save_image_if_path, save_npmat_if_path

@save_npmat_if_path
def combine_wavelet(visible_image, thermal_image):
    coeffs_thermal = pywt.wavedec2(thermal_image, 'haar', level=1)
    coeffs_rgb = pywt.wavedec2(visible_image, 'haar', level=1)

    # Fusión de Coeficientes
    coeffs_fusionados = []
    for c_thermal, c_rgb in zip(coeffs_thermal, coeffs_rgb):
        # Aquí estamos tomando el promedio, pero puedes experimentar con diferentes métodos de fusión.
        coeffs_fusionados.append((np.array(c_thermal) + np.array(c_rgb)) / 2)

    # Reconstrucción de la Imagen
    image = pywt.waverec2(coeffs_fusionados, 'haar')
    return image  

