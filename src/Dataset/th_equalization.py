#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import cv2 as cv

def th_equalization(th_img, thermal_eq):
    
    if thermal_eq.lower() == "clahe":
        clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(6,6))
        return clahe.apply(th_img)
    
    elif thermal_eq.lower() == 'expand':
        # Histogram expansion
        hist, bins = np.histogram(th_img.flatten(), 256, [0,256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        cdf_m = np.ma.masked_equal(cdf, 0) # Avoid zero division
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        return cdf[th_img] # Apply transformation to original image
        
    return th_img