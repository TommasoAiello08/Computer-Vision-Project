#!/usr/bin/env python3
"""Diagnostic script to understand why SIFT matching fails."""
import cv2
import numpy as np
import json

# Load calibration
with open('calibration.json') as f:
    calib = json.load(f)
K = np.array(calib['camera_matrix'], dtype=np.float64)
dist = np.array(calib['dist_coeffs'], dtype=np.float64).flatten()

print('K:')
print(K)
print('dist:', dist)
print()

# Load images
img1 = cv2.imread('Photos/img1.jpeg')
img2 = cv2.imread('Photos/img2.jpeg')
print(f'img1: {img1.shape}')
print(f'img2: {img2.shape}')
print()

# ===== Test on RAW images =====
g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print('=== RAW images (no undistort, no CLAHE) ===')
for nf in [2000, 5000, 10000]:
    sift = cv2.SIFT_create(nfeatures=nf, contrastThreshold=0.04, edgeThreshold=10)
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    inl = 0
    if len(good) >= 8:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.999)
        inl = int(mask.sum()) if mask is not None else 0

    print(f'  nf={nf}: kp1={len(kp1)} kp2={len(kp2)} good={len(good)} inliers={inl}')

# ===== Test on undistorted images =====
print('\n=== Undistorted images ===')
h, w = img1.shape[:2]
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
u1 = cv2.undistort(img1, K, dist, None, newK)
u2 = cv2.undistort(img2, K, dist, None, newK)
g1u = cv2.cvtColor(u1, cv2.COLOR_BGR2GRAY)
g2u = cv2.cvtColor(u2, cv2.COLOR_BGR2GRAY)

for nf in [2000, 5000, 10000]:
    sift = cv2.SIFT_create(nfeatures=nf, contrastThreshold=0.04, edgeThreshold=10)
    kp1, d1 = sift.detectAndCompute(g1u, None)
    kp2, d2 = sift.detectAndCompute(g2u, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    inl = 0
    if len(good) >= 8:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.999)
        inl = int(mask.sum()) if mask is not None else 0

    print(f'  nf={nf}: kp1={len(kp1)} kp2={len(kp2)} good={len(good)} inliers={inl}')

# ===== Test with CLAHE =====
print('\n=== Undistorted + CLAHE ===')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
g1c = clahe.apply(g1u)
g2c = clahe.apply(g2u)

for nf in [2000, 5000, 10000]:
    sift = cv2.SIFT_create(nfeatures=nf, contrastThreshold=0.04, edgeThreshold=10)
    kp1, d1 = sift.detectAndCompute(g1c, None)
    kp2, d2 = sift.detectAndCompute(g2c, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    inl = 0
    if len(good) >= 8:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.999)
        inl = int(mask.sum()) if mask is not None else 0

    print(f'  nf={nf}: kp1={len(kp1)} kp2={len(kp2)} good={len(good)} inliers={inl}')

# ===== Descriptor distance statistics =====
print('\n=== Descriptor distance statistics (RAW, nf=5000) ===')
sift = cv2.SIFT_create(nfeatures=5000)
kp1, d1 = sift.detectAndCompute(g1, None)
kp2, d2 = sift.detectAndCompute(g2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
raw = bf.knnMatch(d1, d2, k=2)
ratios = [m.distance / n.distance for m, n in raw if n.distance > 0]
print(f'  ratio mean={np.mean(ratios):.3f} median={np.median(ratios):.3f}')
print(f'  ratio < 0.7: {sum(1 for r in ratios if r < 0.7)}')
print(f'  ratio < 0.8: {sum(1 for r in ratios if r < 0.8)}')
print(f'  ratio < 0.9: {sum(1 for r in ratios if r < 0.9)}')
print(f'  total pairs: {len(ratios)}')

# ===== Check if dense matching in the script uses wrong R/t =====
# The dense section uses R, t from manual 8-point on 12 points.
# If those are wrong, the E_sift RANSAC will disagree and drop everything.
print('\n=== Cross-check: does dense section E_sift agree with manual E? ===')
# Use the saved inlier points to get the "grid search" F
data = np.load('sift_inliers_best.npz', allow_pickle=True)
pts1_saved = data['points1']
pts2_saved = data['points2']
F_saved = data['F']
print(f'Saved inliers: {len(pts1_saved)}')
print(f'F_saved:\n{F_saved}')

# Now re-do SIFT dense matching (same as script) and see the numbers
sift2 = cv2.SIFT_create(nfeatures=2000)
kp1d, d1d = sift2.detectAndCompute(g1, None)
kp2d, d2d = sift2.detectAndCompute(g2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(d1d, d2d, k=2)

good_m = []
for pair in matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_m.append(m)

print(f'\nDense SIFT (nf=2000, ratio=0.7): {len(good_m)} good matches')

if len(good_m) >= 8:
    p1 = np.float32([kp1d[m.queryIdx].pt for m in good_m])
    p2 = np.float32([kp2d[m.trainIdx].pt for m in good_m])
    E_s, mask_s = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inl_s = int(mask_s.sum()) if mask_s is not None else 0
    print(f'  findEssentialMat inliers: {inl_s}/{len(good_m)}')
    
    if inl_s >= 5:
        p1_inl = p1[mask_s.ravel() == 1]
        p2_inl = p2[mask_s.ravel() == 1]
        _, R_s, t_s, _ = cv2.recoverPose(E_s, p1_inl, p2_inl, K)
        print(f'  R_sift:\n{R_s}')
        print(f'  t_sift: {t_s.ravel()}')
else:
    print('  Too few matches for Essential matrix')

print('\nDone.')
