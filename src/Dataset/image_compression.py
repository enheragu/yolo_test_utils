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
from Dataset.decorators import time_execution_measure, save_image_if_path, save_npmat_if_path

def rescaleChannel(channel, max_value, new_max):
    channel = new_max * (channel / max_value)
    channel = channel.astype(np.uint8)
    return channel

def channelAverage(channel1, channel2):
    channel1 = channel1.astype(np.float64)
    channel2 = channel2.astype(np.float64)
    return rescaleChannel(channel=channel1+channel2, max_value=255+255, new_max=255)

def channelProduct(channel1, channel2):
    channel1 = channel1.astype(np.float64)
    channel2 = channel2.astype(np.float64)
    return rescaleChannel(channel=channel1*channel2, max_value=255*255, new_max=255)


@save_image_if_path   
def combine_hsvt(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image

    # Cast to 32S to avoid saturation when both channels are added
    v = v.astype(np.float64)
    th_channel = th_channel.astype(np.float64)

    intensity = v + th_channel
    _, max_val, _, _ = cv.minMaxLoc(intensity)
    intensity = 255 * (intensity / max_val)
    intensity = intensity.astype(np.uint8)

    hsvt_image = cv.merge([h, s, intensity])
    hsvt_image = cv.cvtColor(hsvt_image, cv.COLOR_HSV2BGR)
    
    return hsvt_image


@save_image_if_path   
def combine_hsvt_v3(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image

    intensity = channelAverage(v, th_channel)

    hsvt_image = cv.merge([h, s, intensity])
    hsvt_image = cv.cvtColor(hsvt_image, cv.COLOR_HSV2BGR)
    
    return hsvt_image

  
@save_image_if_path           
def combine_rgbt(visible_image, thermal_image):
    b,g,r = cv.split(visible_image)
    th_channel = thermal_image
    th_channel = th_channel.astype(np.float64)
    
    for ch in (b,g,r):
        ch = ch.astype(np.float64)
        ch = (ch + th_channel) / 2
        ch = ch.astype(np.uint8)

    rgbt_image = cv.merge([b,g,r])
    
    return rgbt_image


@save_image_if_path
def combine_rgbt_v3(visible_image, thermal_image):
    b,g,r = cv.split(visible_image)
    th_channel = thermal_image
    th_channel = th_channel.astype(np.float64)
    
    for ch in (b,g,r):
        ch = channelAverage(ch, th_channel)

    rgbt_image = cv.merge([b,g,r])
    
    return rgbt_image


@save_image_if_path
def combine_vths(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image
         
    h_shifted = h >> 4
    s_shifted = s >> 4
    hs = h_shifted & (s_shifted << 4)

    vths_image = cv.merge([v,th_channel,hs])
    
    return vths_image


@save_image_if_path
def combine_vths_v2(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV_FULL))
    th_channel = thermal_image
         
    h_shifted = (h // 16) * 16 # Reduce to only 16 values (4 bits) 
    s_shifted = (s // 16) * 16
    hs = h_shifted & (s_shifted << 4)

    vths_image = cv.merge([v,th_channel,hs])
    
    return vths_image


@save_image_if_path
def combine_vths_v3(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV_FULL))
    th_channel = thermal_image
         
    hs = channelAverage(h, s)

    vths_image = cv.merge([v,th_channel,hs])
    
    return vths_image


@save_image_if_path
def combine_vt(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image
    
    averaged = v.astype(np.float64)
    averaged = (averaged + th_channel.astype(np.float64)) / 2
    averaged = averaged.astype(np.uint8)

    vt_image = cv.merge([v,th_channel,averaged])
    
    return vt_image


@save_image_if_path
def combine_vt_v3(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image
    
    both = channelAverage(v, th_channel)
    vt_image = cv.merge([v,th_channel,both])
    
    return vt_image


@save_npmat_if_path
def combine_4ch(visible_image, thermal_image):
    b,g,r = cv.split(visible_image)
    th_channel = thermal_image

    ch4_image = cv.merge([b,g,r,th_channel])

    return ch4_image


@save_npmat_if_path
def combine_lwir_npy(visible_image, thermal_image):
    th_channel = thermal_image
    th_image = cv.merge([th_channel,th_channel,th_channel]).astype(np.uint8)
    return th_channel


@save_npmat_if_path
def combine_vt_2ch(visible_image, thermal_image):
    h,s,v = cv.split(cv.cvtColor(visible_image, cv.COLOR_BGR2HSV))
    th_channel = thermal_image
    vt_image = cv.merge([v,th_channel]).astype(np.uint8)
    return vt_image