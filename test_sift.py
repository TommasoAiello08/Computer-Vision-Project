#!/usr/bin/env python3
"""
Test SIFT feature detection with better error handling
Run this to see if SIFT works on your images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== SIFT TEST SCRIPT ===\n")

# Load images
img1 = cv2.imread('Photos/img1.jpeg')
img2 = cv2.imread('Photos/img2.jpeg')

if img1 is None or img2 is None:
    print("❌ ERROR: Could not load images!")
    exit(1)

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print(f"Image 1: {gray1.shape}, range=[{gray1.min()}, {gray1.max()}]")
print(f"Image 2: {gray2.shape}, range=[{gray2.min()}, {gray2.max()}]")

# Try SIFT with permissive parameters
print("\n--- Testing SIFT with nfeatures=2000, contrastThreshold=0.02 ---")
sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02, edgeThreshold=10)

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"✓ Image 1: {len(kp1)} features")
print(f"✓ Image 2: {len(kp2)} features")

if len(kp1) == 0 or len(kp2) == 0:
    print("\n❌ NO FEATURES DETECTED!")
    print("Trying with even more permissive parameters...")
    
    sift2 = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01, edgeThreshold=20)
    kp1, des1 = sift2.detectAndCompute(gray1, None)
    kp2, des2 = sift2.detectAndCompute(gray2, None)
    print(f"  With contrastThreshold=0.01: {len(kp1)} + {len(kp2)} features")

if len(kp1) > 0 and len(kp2) > 0:
    # Try matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    print(f"\nMatching: {len(matches)} pairs → {len(good_matches)} good matches")
    
    if len(good_matches) >= 8:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
        
        if F is not None and mask is not None:
            n_inliers = np.sum(mask.ravel() == 1)
            print(f"✓ RANSAC inliers: {n_inliers}/{len(good_matches)}")
            
            if n_inliers >= 50:
                print("\n✅ SUCCESS! SIFT works well on these images")
            elif n_inliers >= 15:
                print("\n⚠️  MODERATE: SIFT works but not optimal")
            else:
                print("\n❌ POOR: Too few reliable matches")
        else:
            print("\n❌ Could not compute fundamental matrix")
    else:
        print(f"\n❌ Only {len(good_matches)} matches (need at least 8)")
else:
    print("\n❌ SIFT FAILED completely on these images")
    print("Recommendation: Use manual point annotation")

print("\n" + "="*60)
print("Copy these parameters to your notebook if SIFT worked:")
print("  sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)")
print("="*60)
