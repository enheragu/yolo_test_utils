#!/usr/bin/env python3
# encoding: utf-8
"""
    Creates different approachs of mixing RGB with Thermal images
    hsvt -> combines v + t and reescalate that channel
    rgbt -> averages each chanel with thermal data (r+t/2)
    4ch -> Stores ndarray with all foru channels as [b,g,r,t]
    vths -> Intensiti from visible, Thermal and HS compressed in one channel (4 bits of each channel).
        It seems that most of the relevant information is on Intensity channels, and pretty little in the color part. 
        So color is compressed in one channel.
        4 bits shifted to the right (same as 16 division, but faster)
        8 7 6 5 4 3 2 1 -> Keeps from 8 to 5 bits from both images
        8h 7h 6h 5h 8v 7v 6v 5v -> shifts one of the channels back and adds them
    vt -> Removes color information having one channel for Intensity from visual image, Thermal channel and the average of both in the third channel.
"""
import numpy as np
import cv2 as cv

from utils import log, bcolors

def combine_hsvt(visible_image, thermal_image, path):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = cv.cvtColor(thermal_image, cv.COLOR_BGR2GRAY)

    # Cast to 32S to avoid saturation when both channels are added
    v = v.astype(np.float64)
    th_channel = th_channel.astype(np.float64)

    intensity = v + th_channel
    _, max_val, _, _ = cv.minMaxLoc(intensity)
    intensity = 255 * (intensity / max_val)
    intensity = intensity.astype(np.uint8)

    hsvt_image = cv.merge([h, s, intensity])
    hsvt_image = cv.cvtColor(hsvt_image, cv.COLOR_HSV2BGR)
    
    cv.imwrite(path, hsvt_image)
    return hsvt_image

              
def combine_rgbt(visible_image, thermal_image, path):
    b,g,r = cv.split(visible_image)
    th_channel = cv.cvtColor(thermal_image, cv.COLOR_BGR2GRAY)
    th_channel = th_channel.astype(np.float64)
    
    for ch in (b,g,r):
        ch = ch.astype(np.float64)
        ch = (ch + th_channel) / 2
        ch = ch.astype(np.uint8)

    rgbt_image = cv.merge([b,g,r])
    
    cv.imwrite(path, rgbt_image)
    return rgbt_image


def combine_4ch(visible_image, thermal_image, path):
    b,g,r = cv.split(visible_image)
    th_channel = cv.cvtColor(thermal_image, cv.COLOR_BGR2GRAY)

    ch4_image = cv.merge([b,g,r,th_channel])

    # cv.imwrite(path, ch4_image)
    np.save(path.replace('.png',''), ch4_image)
    # np.savez_compressed(path.replace('.png',''), image = ch4_image)
    return ch4_image


def combine_vths(visible_image, thermal_image, path):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = cv.cvtColor(thermal_image, cv.COLOR_BGR2GRAY)
         
    h_shifted = h >> 4
    s_shifted = s >> 4
    hs = h_shifted & (s_shifted << 4)

    # print(f"{v.shape =}; {th_channel.shape =}; {hs.shape =}; ")
    vths_image = cv.merge([v,th_channel,hs])
    
    cv.imwrite(path, vths_image)
    return vths_image


def combine_vt(visible_image, thermal_image, path):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = cv.cvtColor(thermal_image, cv.COLOR_BGR2GRAY)
    
    averaged = v.astype(np.float64)
    averaged = (averaged + th_channel.astype(np.float64)) / 2
    averaged = averaged.astype(np.uint8)

    vt_image = cv.merge([v,th_channel,averaged])
    
    cv.imwrite(path, vt_image)
    return vt_image


def combine_lwir_1ch(visible_image, thermal_image, path):
    th_channel = cv.cvtColor(thermal_image, cv.COLOR_BGR2GRAY)
    np.save(path.replace('.png',''), th_channel)
    return th_channel
