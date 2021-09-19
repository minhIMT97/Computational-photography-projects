# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:11:08 2021

@author: Admin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

def edge_detection(img):
    blur = cv2.GaussianBlur(img,(11,11),5)
    gx, gy = np.gradient(blur)
    
    edge = np.sqrt(gx**2 + gy**2)
    
    edge = cv2.Canny(img,50,150)
    
    return edge

def hough_transform(edge, threshold, rhoRes, thetaRes):
    H = np.zeros((thetaRes, rhoRes))
    rhoScale = np.zeros((rhoRes, ))
    for x in range(edge.shape[1]):
        for y in range(edge.shape[0]):
            if edge[y,x] > threshold:
                for theta in range(thetaRes):
                    rho = x*np.cos(np.pi*theta/180) + y*np.sin(np.pi*theta/180)
                    H[theta, rho] += 1

img = cv2.imread('data/img01.jpg',0)
edge = edge_detection(img)
cv2.imshow('edge', edge)
cv2.waitKey(0)