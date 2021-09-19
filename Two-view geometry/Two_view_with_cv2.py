# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 21:25:01 2021

@author: Binh Minh
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    return M2s

def triangulate(P1, pts1, P2, pts2):

    A = np.zeros((4, 4))
    X = np.zeros((pts1.shape[0], 3))
    i = 0
    for p1, p2 in zip(pts1, pts2):
        # p1_homo = np.concatenate((p1, [1]), axis = 0)
        # p2_homo = np.concatenate((p1, [1]), axis = 0)
        # Camera 1
        
        A[0] = p1[1]*P1[2] - P1[1]
        A[1] = P1[0] - p1[0]*P1[2]
        
        # Camera 2
        
        A[2] = p2[1]*P2[2] - P2[1]
        A[3] = P2[0] - p2[0]*P2[2]
        
        # SVD
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        x = vh[-1, :]
        
        x /= x[-1]
        
        X[i] = x[0:3]
        i+=1
  
    return X

# Load images
img1 = cv2.imread('data/im1.png',0)
img2 = cv2.imread('data/im2.png',0)

# Find key points with SIFT
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Draw matches
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good, None,flags=2)

# plt.imshow(img3),plt.show()

# compute F
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
print(F)

# Load camera matrices
cam_mat = np.load('data/intrinsics.npz')
K1 = cam_mat['K1']
K2 = cam_mat['K2']

# Compute Essential Matrix
E = K2.T.dot(F).dot(K1)
print(E)

# First camera matrix
P1 = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0]])
P1 = np.matmul(K1, P1)

# 4 possibles second camera matrices estimated from Essential Matrix
P2 = camera2(E)
print(P2)

# Take one of them to do triangulation. The valid one reconstructs 3D points
# with possitive Z values
P2_i = np.matmul(K2,P2[:,:,2])

# Triangulation
X = triangulate(P1, pts1, P2_i, pts2)
print(X)

# Plot 3D points
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X[:,0], X[:,1], X[:,2])
plt.show()

