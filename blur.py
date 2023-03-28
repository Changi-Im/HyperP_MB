import cv2
import numpy as np
import glob
import os
import torch

def generate_mask(img, sigma, type='none'):
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = img.astype('float32')
    mask = ((img - np.min(img))/(np.max(img)-np.min(img)))
    if type == 'high':
        mask = 1-mask
    
    return mask

def generate_mask_(img1, img2, target, sigma, type='none'):
    img1 = cv2.GaussianBlur(img1, (0, 0), sigma)
    img2 = cv2.GaussianBlur(img2, (0, 0), sigma)
    target = cv2.GaussianBlur(target, (0, 0), sigma)
    
    img1 = img1.astype('float32')
    img2 = img2.astype('float32')
    target = target.astype('float32')
    
    # subtraction
    sub1 = target - img1
    sub2 = target - img2
    
    # addition
    img = sub1/2 + sub2/2
    
    # normalization
    mask1 = ((img1 - np.min(img1))/(np.max(img1)-np.min(img1)))
    mask2 = 1 - (((img2 - np.min(img2))/(np.max(img2)-np.min(img2))))
    maskt = 1- (((img - np.min(img))/(np.max(img)-np.min(img))))
    
    return mask1, mask2, maskt
   