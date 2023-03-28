import numpy as np
import cv2
import torch
import AGC

def homography_warp(source1_path, source2_path):  # matching through SIFT algorithm
    MIN_MATCH_COUNT = 20
    source1 = cv2.imread(source1_path)
    source2 = cv2.imread(source2_path)
    
    # Adaptive gamma correction
    img1_g = cv2.cvtColor(source1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(source2, cv2.COLOR_BGR2GRAY)
    
    img1 = AGC.AGC(img1_g)
    img2 = AGC.AGC(img2_g)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img2, None)
    kp2, des2 = sift.detectAndCompute(img1, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        
    else:
        print('Not enough matches are found - {}/{}'.format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        source1 = cv2.cvtColor(source1, cv2.COLOR_BGR2Lab)
        source2 = cv2.cvtColor(source2, cv2.COLOR_BGR2Lab)
        source1_g, a1, b1 = cv2.split(source1)
        source2_g, a2, b2 = cv2.split(source2)
        
        source1_g = np.reshape(source1_g, [1, 1, source1_g.shape[0], source1_g.shape[1]])  
        source2_g = np.reshape(source2_g, [1, 1, source2_g.shape[0], source2_g.shape[1]])  
        source1_g = torch.from_numpy(source1_g).float()
        source2_g = torch.from_numpy(source2_g).float()
        
        return source1_g, source2_g, a1, a2, b1, b2
        

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    
    aligned_source1 = np.zeros_like(source1,dtype='uint8')
    for i in range(3):
        aligned_source1[...,i] = cv2.warpPerspective(source1[...,i], M, (source1.shape[1], source1.shape[0]), flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP)
        
    aligned_source1 = cv2.cvtColor(aligned_source1, cv2.COLOR_BGR2Lab)
    source2 = cv2.cvtColor(source2, cv2.COLOR_BGR2Lab)
    aligned_source1_g, a1, b1 = cv2.split(aligned_source1)
    source2_g, a2, b2 = cv2.split(source2)
    
    aligned_source1_g = np.reshape(aligned_source1_g, [1, 1, aligned_source1_g.shape[0], aligned_source1_g.shape[1]])  
    source2_g = np.reshape(source2_g, [1, 1, source2_g.shape[0], source2_g.shape[1]])  
    aligned_source1_g = torch.from_numpy(aligned_source1_g).float()
    source2_g = torch.from_numpy(source2_g).float()

    return aligned_source1_g, source2_g, a1, a2, b1, b2
