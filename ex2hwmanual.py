# %% [markdown]
# # Ex.2 Eight point algorithm for Essential Matrix estimation

# %%
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: save to file instead of displaying
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormaps
import random
import numpy.linalg as la
import os

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ### Step 1: Load images and calibration data
# 
# Load the calibration matrix from the JSON file

# %%
with open('calibration.json', 'r') as f:
    calib = json.load(f)

# Extract Camera Intrinsic Matrix K
K = np.array(calib['camera_matrix'])

# Load images from Photos folder (converted to RGB for plotting)
img1 = cv2.imread('Photos/img1.jpeg')
img2 = cv2.imread('Photos/img2.jpeg')

if img1 is None or img2 is None:
    raise FileNotFoundError("Could not load images from Photos/ folder")

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

print(f"Image 2 Size: {img2_rgb.shape[1]}x{img2_rgb.shape[0]}")
print(f"Image 1 Size: {img1_rgb.shape[1]}x{img1_rgb.shape[0]}")

# %% [markdown]
# ### Step 2a: TEST - Automatic Feature Detection with SIFT (Optional)
# 
# Let's test SIFT to see how many and which correspondences it can find automatically.
# This is a **preview** to decide if we want to use automatic matching or add more manual points.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ---------------------------------------------------------------------
# FAST PATH: usa risultati salvati per evitare di rilanciare grid-search
# ---------------------------------------------------------------------
USE_SAVED = True
SAVED_PATH = 'sift_inliers_best.npz'

if USE_SAVED and Path(SAVED_PATH).exists():
    data = np.load(SAVED_PATH, allow_pickle=True)
    points1 = data['points1'].astype(np.float32)
    points2 = data['points2'].astype(np.float32)
    F = data['F'].astype(np.float64)

    print('Caricati risultati salvati da:', SAVED_PATH)
    print(f'Inliers caricati: {len(points1)}')

    # Visualizzazione robusta delle corrispondenze (senza drawMatches)
    img1 = cv2.imread('Photos/img1.jpeg')
    img2 = cv2.imread('Photos/img2.jpeg')
    if img1 is None or img2 is None:
        raise FileNotFoundError('Immagini non trovate in Photos/')

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(img1_rgb)
    axes[0].scatter(points1[:, 0], points1[:, 1], s=8, c='lime')
    axes[0].set_title(f'Image 1 - Inlier points ({len(points1)})')
    axes[0].axis('off')

    axes[1].imshow(img2_rgb)
    axes[1].scatter(points2[:, 0], points2[:, 1], s=8, c='cyan')
    axes[1].set_title(f'Image 2 - Inlier points ({len(points2)})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_sift_inliers_saved.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {OUTPUT_DIR}/01_sift_inliers_saved.png')

else:
    # -----------------------------
    # FULL COMPUTE (più lento)
    # -----------------------------
    # Caricamento calibrazione
    with open('calibration.json', 'r') as f:
        calib = json.load(f)

    K = np.array(calib['camera_matrix'], dtype=np.float64)
    dist = np.array(calib['dist_coeffs'], dtype=np.float64).flatten()

    # Caricamento immagini
    img1 = cv2.imread('Photos/img1.jpeg')
    img2 = cv2.imread('Photos/img2.jpeg')
    if img1 is None or img2 is None:
        raise FileNotFoundError('Immagini non trovate in Photos/')

    # Undistort
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    newK1, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w1, h1), 1, (w1, h1))
    newK2, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w2, h2), 1, (w2, h2))
    u1 = cv2.undistort(img1, K, dist, None, newK1)
    u2 = cv2.undistort(img2, K, dist, None, newK2)

    # Grayscale + CLAHE
    g1 = cv2.cvtColor(u1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(u2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(g1)
    g2 = clahe.apply(g2)

    # Maschera anti-bordi neri dovuti a undistort
    mask1 = (cv2.cvtColor(u1, cv2.COLOR_BGR2GRAY) > 8).astype(np.uint8) * 255
    mask2 = (cv2.cvtColor(u2, cv2.COLOR_BGR2GRAY) > 8).astype(np.uint8) * 255

    def sampson_errors(F, pts1, pts2):
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1), dtype=np.float64)])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1), dtype=np.float64)])

        Fx1 = (F @ pts1_h.T).T
        Ftx2 = (F.T @ pts2_h.T).T
        x2tFx1 = np.sum(pts2_h * Fx1, axis=1)

        denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
        denom = np.maximum(denom, 1e-12)
        return (x2tFx1 ** 2) / denom

    def knn_matches(des1, des2, matcher_type='bf'):
        if matcher_type == 'flann':
            index_params = dict(algorithm=1, trees=8)  # KDTree
            search_params = dict(checks=80)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        return matcher.knnMatch(des1, des2, k=2)

    def unique_matches(good_matches):
        # Evita fan-out: un match per queryIdx e un match per trainIdx
        good_sorted = sorted(good_matches, key=lambda m: m.distance)
        used_q = set()
        used_t = set()
        unique = []
        for m in good_sorted:
            if m.queryIdx in used_q or m.trainIdx in used_t:
                continue
            unique.append(m)
            used_q.add(m.queryIdx)
            used_t.add(m.trainIdx)
        return unique

    def run_sift_pipeline(g1, g2, mask1, mask2, nfeatures=4000, contrast=0.01, edge=20, ratio=0.8, ransac_thr=2.0, matcher_type='bf'):
        sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=contrast,
            edgeThreshold=edge
        )

        kp1, des1 = sift.detectAndCompute(g1, mask1)
        kp2, des2 = sift.detectAndCompute(g2, mask2)
        if des1 is None or des2 is None:
            return None

        raw = knn_matches(des1, des2, matcher_type=matcher_type)

        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        good = unique_matches(good)

        if len(good) < 8:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.FM_RANSAC
        F, maskF = cv2.findFundamentalMat(pts1, pts2, method, ransac_thr, 0.999)
        if F is None or maskF is None:
            return None

        inliers = maskF.ravel().astype(bool)
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]
        if pts1_in.shape[0] < 8:
            return None

        errs = sampson_errors(F, pts1_in.astype(np.float64), pts2_in.astype(np.float64))
        med_err = float(np.median(errs))
        mean_err = float(np.mean(errs))

        score = pts1_in.shape[0] / (1.0 + med_err)

        return {
            'kp1': kp1,
            'kp2': kp2,
            'good': good,
            'inliers_mask': inliers,
            'pts1_in': pts1_in,
            'pts2_in': pts2_in,
            'F': F,
            'num_kp1': len(kp1),
            'num_kp2': len(kp2),
            'num_good': len(good),
            'num_inliers': int(np.sum(inliers)),
            'median_sampson': med_err,
            'mean_sampson': mean_err,
            'score': score,
            'params': (nfeatures, contrast, edge, ratio, ransac_thr, matcher_type)
        }

    best = None
    grid = {
        'nfeatures': [4000, 8000],
        'contrast': [0.02, 0.008],
        'edge': [20, 35],
        'ratio': [0.82, 0.88, 0.92],
        'ransac_thr': [2.0, 3.0, 4.0],
        'matcher_type': ['bf', 'flann']
    }

    for nf in grid['nfeatures']:
        for ct in grid['contrast']:
            for ed in grid['edge']:
                for rr in grid['ratio']:
                    for rt in grid['ransac_thr']:
                        for mt in grid['matcher_type']:
                            out = run_sift_pipeline(g1, g2, mask1, mask2, nf, ct, ed, rr, rt, mt)
                            if out is None:
                                continue
                            if best is None or out['score'] > best['score']:
                                best = out

    if best is None:
        raise RuntimeError('Nessuna configurazione valida trovata con SIFT + RANSAC')

    points1 = best['pts1_in']
    points2 = best['pts2_in']
    F = best['F']

    print('Best params (nfeatures, contrast, edge, ratio, ransac_thr, matcher):', best['params'])
    print(f"Keypoints: {best['num_kp1']} / {best['num_kp2']}")
    print(f"Good matches (unique): {best['num_good']}")
    print(f"Inliers finali: {best['num_inliers']}")
    print(f"Sampson median: {best['median_sampson']:.4f}")
    print(f"Sampson mean:   {best['mean_sampson']:.4f}")

    # Visualizzazione inlier senza artefatti di fan-out
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(img1_rgb)
    axes[0].scatter(points1[:, 0], points1[:, 1], s=8, c='lime')
    axes[0].set_title(f'Image 1 - Inlier points ({len(points1)})')
    axes[0].axis('off')

    axes[1].imshow(img2_rgb)
    axes[1].scatter(points2[:, 0], points2[:, 1], s=8, c='cyan')
    axes[1].set_title(f'Image 2 - Inlier points ({len(points2)})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_sift_inliers_computed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {OUTPUT_DIR}/01_sift_inliers_computed.png')

    np.savez(SAVED_PATH, points1=points1, points2=points2, F=F)


# (TEST block removed: referenced undefined kp1_test/des1_test variables.
#  All functionality is covered by the SIFT grid-search above and the dense
#  reconstruction section below.)

# %% [markdown]
# ### Step 2b: Manual correspondences (for algorithm validation)

# %%


# # 'Immagine storta' -> Coordinates for pts2
# immagine_storta = np.array([
#     (1531, 1654), (1878, 1651), (2203, 1622), (2305, 1629), (2349, 1973),
#     (2593, 1634), (2600, 1973), (2726, 1716), (2872, 1979), (2848, 1851),
#     (2968, 1856), (3295, 1884), (3425, 1846), (3558, 1716), (3456, 1979),
#     (3725, 1979), (3934, 1721), (3991, 1981), (3885, 1702), (4093, 1657),
#     (3158, 1168), (4659, 2241)
# ], dtype=np.float32)

# # 'Immagine dritta' -> Coordinates for pts1
# immagine_dritta = np.array([
#     (1284, 1675), (1515, 1687), (1983, 1669), (2281, 1676), (2306, 2052),
#     (2289, 1695), (2298, 2058), (2633, 1778), (2652, 2040), (2843, 1916),
#     (2832, 1918), (3247, 1958), (3193, 1912), (3444, 1796), (3448, 2053),
#     (3618, 2053), (3657, 1844), (3731, 2052), (3948, 1786), (3942, 1749),
#     (2268, 1245), (3891, 2287)
# ], dtype=np.float32)

# 'Immagine storta' -> Coordinates for pts2
immagine_storta = np.array([
    (1531, 1654), (2203, 1622), (2305, 1629),
    (2726, 1716), (2872, 1979), (2848, 1851),
    (3425, 1846), (3456, 1979),
    (3934, 1721), (3885, 1702),
    (3158, 1168), (4659, 2241)
], dtype=np.float32)

# 'Immagine dritta' -> Coordinates for pts1
immagine_dritta = np.array([
    (1284, 1675), (1983, 1669), (2281, 1676),
    (2633, 1778), (2652, 2040), (2843, 1916),
    (3193, 1912), (3448, 2053),
    (3657, 1844), (3948, 1786),
    (2268, 1245), (3891, 2287)
], dtype=np.float32)

# Verify counts match
assert(immagine_storta.shape == immagine_dritta.shape), "Points array shapes must match!"

# Assign to standard variable names used in subsequent steps
pts2 = immagine_storta
pts1 = immagine_dritta

print(f"Loaded {len(pts1)} manual correspondences.")

# # --- ESTIMATE FUNDAMENTAL MATRIX F ---
# # Since these are trusted manual points and we have > 8, we use least squares estimation (FM_8POINT).
# F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

# print("\nFundamental Matrix F (computed from manual points):")
# print(F)

# %% [markdown]
# ### Step 3: Visualize manual points with colors

# %%
### NEW CELL: VISUALIZE MANUAL POINTS WITH COLORS ###

def show_manual_points(img1, img2, p1, p2):
    """Plots points side-by-side with colors and numbering."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    axes[0].imshow(img1)
    axes[0].set_title('Image 1 (Straight image) - Annotated Points')
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title('Image 2 (Rotated image) - Annotated Points')
    axes[1].axis('off')
    
    n = len(p1)
    # Generate a set of distinctive colors based on the number of points
    colors = cm.rainbow(np.linspace(0, 1, n))
    
    for i in range(n):
        # Image 1 points
        x1, y1 = p1[i]
        axes[0].plot(x1, y1, marker='o', color=colors[i], markersize=6)
        axes[0].text(x1 + 5, y1 - 10, str(i), color=colors[i], fontsize=12, weight='bold')
        
        # Image 2 points
        x2, y2 = p2[i]
        axes[1].plot(x2, y2, marker='o', color=colors[i], markersize=6)
        axes[1].text(x2 + 5, y2 - 10, str(i), color=colors[i], fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_manual_points.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {OUTPUT_DIR}/02_manual_points.png')

# Show the visualization
show_manual_points(img1_rgb, img2_rgb, pts1, pts2)

# %% [markdown]
# ### Step 4: From fundamental matrix to essential matrix

# %%
### 8-POINT ALGORITHM FOR F AND E ESTIMATION

def convert_to_homo(pts):
    ''' Converts 2D points to homogeneous coordinates (x, y, 1) '''
    return np.array([(*point, 1.0) for point in pts])

def norm_matrix(pts):
    ''' Computes the normalization matrix T for the 8-point algorithm '''
    pts = np.float64(pts)
    # Compute centroid (mean) of the points
    mean = np.mean(pts, axis=0)
    
    # Compute the average distance from the centroid
    # We want the average distance to be sqrt(2)
    scale = np.mean(la.norm(pts - mean, axis=1)) / np.sqrt(2.0)

    # Transformation matrix to shift origin to mean and scale appropriately
    T = np.array([
        [1.0 / scale, 0.0, -mean[0] / scale],
        [0.0, 1.0 / scale, -mean[1] / scale],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return T

def eight_point_algorithm(pts1, pts2):
    ''' Estimates the Fundamental Matrix using the Normalized 8-Point Algorithm '''
    n = len(pts1)
    
    # STEP 1: Normalize points
    T1 = norm_matrix(pts1)
    T2 = norm_matrix(pts2)
    
    pts1_homo = convert_to_homo(pts1)
    pts2_homo = convert_to_homo(pts2)
    
    # Apply normalization (T * p)
    pts1_norm = (T1 @ pts1_homo.T).T
    pts2_norm = (T2 @ pts2_homo.T).T
    
    # STEP 2: Build the linear system matrix A
    A = np.zeros((n, 9))
    for i in range(n):
        u1, v1, _ = pts1_norm[i]
        u2, v2, _ = pts2_norm[i]
        A[i] = [u2*u1, u2*v1, u2,
                v2*u1, v2*v1, v2,
                u1,    v1,    1]
        
    # STEP 3: Solve Af = 0 using SVD
    U, S, Vt = la.svd(A)
    # The solution is the column of V (or row of V transposed) corresponding to the smallest singular value
    F_norm = Vt[-1, :].reshape(3, 3)
    
    # STEP 4: Enforce the Rank-2 constraint on F
    # Set smallest singular value to 0 to ensure rank(F) = 2
    U_f, S_f, Vt_f = la.svd(F_norm)
    S_f_corrected = S_f.copy()
    S_f_corrected[-1] = 0  # Force rank-2
    F_rank2 = U_f @ np.diag(S_f_corrected) @ Vt_f
    
    # STEP 5: Denormalize to get the final F
    F = T2.T @ F_rank2 @ T1
    
    # Normalize by Frobenius norm (more numerically stable than F[2,2])
    F = F / np.linalg.norm(F)
    
    return F

# Compute Matrices

# 1. Compute Fundamental Matrix using our custom function
F = eight_point_algorithm(pts1, pts2)

print("Fundamental Matrix F (Custom 8-Point):")
print(F)

# 2. Compute Essential Matrix E from F and the intrinsic matrix K
E = K.T @ F @ K

print("\nEssential Matrix E:")
print(E)

# 3. Recover relative Rotation (R) and Translation (t) using OpenCV 
# (Pose recovery from E involves complex Cheirality checks, so keeping cv2 here is standard)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

print("\nRotation Matrix R:")
print(R)

print("\nTranslation Vector t:")
print(t)

# %% [markdown]
# ### Comparison with OpenCV Five-Point Algorithm
# 
# The assignment requires comparing our 8-point algorithm with OpenCV's 5-point algorithm.
# The five-point algorithm directly computes the Essential Matrix E for calibrated cameras.

# %%
# Compute Essential Matrix using OpenCV's five-point algorithm with RANSAC
E_opencv, mask_opencv = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, 
                                              prob=0.999, threshold=1.0)

print("\n=== COMPARISON: Custom 8-Point vs OpenCV 5-Point ===")
print("\nEssential Matrix E (Custom 8-Point):")
print(E)
print("\nEssential Matrix E (OpenCV 5-Point with RANSAC):")
print(E_opencv)

# Recover pose from OpenCV's Essential Matrix
_, R_opencv, t_opencv, mask_pose_opencv = cv2.recoverPose(E_opencv, pts1, pts2, K)

print("\n--- Rotation Matrix Comparison ---")
print("R (Custom):")
print(R)
print("\nR (OpenCV):")
print(R_opencv)
print(f"\nRotation difference (Frobenius norm): {la.norm(R - R_opencv):.6f}")

print("\n--- Translation Vector Comparison ---")
print("t (Custom):")
print(t.ravel())
print("\nt (OpenCV):")
print(t_opencv.ravel())
print(f"\nTranslation angular difference: {np.arccos(np.clip(np.dot(t.ravel(), t_opencv.ravel()), -1, 1)) * 180 / np.pi:.4f} degrees")

# Count inliers from RANSAC
inliers_count = np.sum(mask_opencv)
print(f"\n--- RANSAC Inliers ---")
print(f"Inliers: {inliers_count}/{len(pts1)} points")
print(f"Outliers: {len(pts1) - inliers_count} points")

if inliers_count < len(pts1):
    print("\nNote: RANSAC detected outliers. With manual annotation, this suggests")
    print("measurement errors or poor feature localization in some correspondences.")

# %% [markdown]
# ### Discussion: 8-Point vs 5-Point Algorithm
# 
# **Key Differences:**
# 
# 1. **Minimal Set**: 
#    - 5-point algorithm: Requires only 5 correspondences (minimal)
#    - 8-point algorithm: Requires at least 8 correspondences
# 
# 2. **Direct Computation**:
#    - 5-point: Directly computes Essential matrix E (for calibrated cameras)
#    - 8-point: Computes Fundamental matrix F, then E = K^T F K
# 
# 3. **Robustness**:
#    - 5-point with RANSAC: Automatically handles outliers
#    - 8-point: Assumes all correspondences are correct (least squares)
# 
# 4. **Accuracy**:
#    - With perfect correspondences: Similar results
#    - With noise/outliers: 5-point + RANSAC is more robust
#    - With many good points: 8-point can be more stable
# 
# **Expected Results:**
# - If all manual correspondences are accurate: Results should be very similar
# - If some correspondences have errors: OpenCV may flag them as outliers
# - The Essential Matrix should satisfy the same epipolar constraint in both cases

# %%
# # We pass the inlier points so the algorithm can verify the correct physical pose (points must be in front of the camera)
# _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)

# print("\nRotation Matrix R:")
# print(R)
# print("\nTranslation Vector t:")
# print(t)


# %% [markdown]
# ### Step 5: Computer epipolar error

# %%
def convert_to_homo(pts):
    ''' Converts 2D points to homogeneous coordinates. '''
    return [(*point, 1) for point in pts]

def compute_epipolar_error(pts1, pts2, F):
    pts1_homo = np.array(convert_to_homo(pts1))
    pts2_homo = np.array(convert_to_homo(pts2))
    n = len(pts1)
    errors = np.zeros(n)
    
    for i in range(n):
        line2 = F @ pts1_homo[i]
        line1 = F.T @ pts2_homo[i]
        
        num1 = abs(pts1_homo[i].T @ F.T @ pts2_homo[i])
        num2 = abs(pts2_homo[i].T @ F @ pts1_homo[i])
        
        denom1 = np.sqrt(line1[0]**2 + line1[1]**2)
        denom2 = np.sqrt(line2[0]**2 + line2[1]**2)
        
        d1 = num1 / denom1
        d2 = num2 / denom2
        errors[i] = (d1 + d2) / 2  # average of distances

    return errors.mean(), errors

# Pass the manual points to calculate the error
mean_error, errors = compute_epipolar_error(pts1, pts2, F)

print(f"\nMean epipolar distance: {mean_error:.6f} pixels")
print(f"Max epipolar distance: {errors.max():.6f} pixels")
print(f"Min epipolar distance: {errors.min():.6f} pixels")

# %% [markdown]
# ### Step 6: Plot epipolar lines

# %%
def plot_epipolar_lines(img1, img2, pts1, pts2, F):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    n = len(pts1)
    axes[0].imshow(img1)
    axes[0].set_title('Image 1')
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].set_title('Image 2')
    axes[1].axis('off')

    pts1_homo = np.array(convert_to_homo(pts1))
    pts2_homo = np.array(convert_to_homo(pts2))

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    def plot_line_safely(ax, w, h, line, color='r'):
        a, b, c = line
        if abs(a) < 1e-10 and abs(b) < 1e-10:
            return False

        x_intersects, y_intersects = [], []

        # Intersection with y=0 (top edge): a*x + c = 0 => x = -c/a
        if abs(a) > 1e-10:
            x_top = -c / a
            if 0 <= x_top <= w:
                x_intersects.append(x_top); y_intersects.append(0)

        # Intersection with y=h (bottom edge): a*x + b*h + c = 0 => x = -(b*h+c)/a
        if abs(a) > 1e-10:
            x_bottom = -(b * h + c) / a
            if 0 <= x_bottom <= w:
                x_intersects.append(x_bottom); y_intersects.append(h)

        y_left = -c / b
        if 0 <= y_left <= h:
            x_intersects.append(0); y_intersects.append(y_left)

        y_right = -(a * w + c) / b
        if 0 <= y_right <= h:
            x_intersects.append(w); y_intersects.append(y_right)

        if len(x_intersects) >= 2:
            ax.plot(x_intersects, y_intersects, color, linewidth=0.4)
            return True
        return False

    # Plot lines for IMAGE 2
    for i in range(n):
        line = F @ pts1_homo[i]
        plot_line_safely(axes[1], w2, h2, line, 'r')

    # Plot lines for IMAGE 1
    for i in range(n):
        line = F.T @ pts2_homo[i]
        plot_line_safely(axes[0], w1, h1, line, 'b')

    # Plot points
    axes[0].plot(np.array(pts1)[:, 0], np.array(pts1)[:, 1], 'go', markersize=2)
    axes[1].plot(np.array(pts2)[:, 0], np.array(pts2)[:, 1], 'go', markersize=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_epipolar_lines.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {OUTPUT_DIR}/03_epipolar_lines.png')

# We pass all manual points directly to the plotting function
plot_epipolar_lines(img1_rgb, img2_rgb, pts1, pts2, F)

# %% [markdown]
# ## Part 3: 3D Reconstruction and Visualization (2 pts)
# 
# In this section we:
# 1. Use the recovered camera poses (R, t) to triangulate the corresponding points
# 2. Reconstruct the 3D structure of the scene
# 3. Visualize the reconstructed 3D points from multiple viewpoints
# 
# The camera matrices are:
# - **Camera 1 (reference)**: P1 = K [I | 0] (identity pose)
# - **Camera 2 (relative)**: P2 = K [R | t] (recovered from Essential matrix)

# %%
# Build projection matrices for both cameras
# Camera 1 is at the origin (world reference frame)
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

# Camera 2 has the relative pose [R | t]
P2 = K @ np.hstack([R, t])

print("Projection Matrix P1 (Camera 1 - Reference):")
print(P1)
print("\nProjection Matrix P2 (Camera 2 - Relative pose):")
print(P2)

# Triangulate the corresponding points
# cv2.triangulatePoints expects 2xN arrays of image points
pts1_for_triangulation = pts1.T  # Shape: (2, N)
pts2_for_triangulation = pts2.T  # Shape: (2, N)

# Triangulate: returns 4xN homogeneous coordinates
points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1_for_triangulation, pts2_for_triangulation)

# Convert from homogeneous to 3D Cartesian coordinates
points_3d = points_4d_homogeneous[:3, :] / points_4d_homogeneous[3, :]
points_3d = points_3d.T  # Shape: (N, 3)

print(f"\n3D Points Reconstructed: {points_3d.shape[0]} points")
print("\nFirst 5 reconstructed 3D points:")
print(points_3d[:5])

# Check reconstruction quality: reproject points back to images
def reproject_points(points_3d, P):
    """Reproject 3D points to image using projection matrix P"""
    points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    projected_homo = (P @ points_3d_homo.T).T
    projected_2d = projected_homo[:, :2] / projected_homo[:, 2:3]
    return projected_2d

reprojected_pts1 = reproject_points(points_3d, P1)
reprojected_pts2 = reproject_points(points_3d, P2)

# Compute reprojection errors
reproj_error_1 = np.mean(np.linalg.norm(pts1 - reprojected_pts1, axis=1))
reproj_error_2 = np.mean(np.linalg.norm(pts2 - reprojected_pts2, axis=1))

print(f"\nReprojection Error Camera 1: {reproj_error_1:.4f} pixels")
print(f"Reprojection Error Camera 2: {reproj_error_2:.4f} pixels")
print(f"Average Reprojection Error: {(reproj_error_1 + reproj_error_2) / 2:.4f} pixels")

# %%
# 3D Visualization of the reconstructed scene
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 12))

# We'll create 4 different viewpoints of the 3D reconstruction
viewpoints = [
    (30, 45, "View 1: Elevation 30°, Azimuth 45°"),
    (20, 135, "View 2: Elevation 20°, Azimuth 135°"),
    (60, -45, "View 3: Elevation 60°, Azimuth -45°"),
    (10, 90, "View 4: Elevation 10°, Azimuth 90°")
]

for idx, (elev, azim, title) in enumerate(viewpoints, 1):
    ax = fig.add_subplot(2, 2, idx, projection='3d')
    
    # Plot the reconstructed 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               c='red', marker='o', s=100, label='3D Points', alpha=0.8)
    
    # Annotate points with their index
    for i, point in enumerate(points_3d):
        ax.text(point[0], point[1], point[2], f'{i}', fontsize=8)
    
    # Plot camera centers
    # Camera 1 is at origin
    cam1_center = np.array([0, 0, 0])
    ax.scatter(*cam1_center, c='blue', marker='^', s=200, label='Camera 1')
    
    # Camera 2 center: C2 = -R^T * t
    cam2_center = -R.T @ t
    ax.scatter(*cam2_center.ravel(), c='green', marker='^', s=200, label='Camera 2')
    
    # Draw camera optical axes
    axis_length = np.max(np.abs(points_3d)) * 0.3
    
    # Camera 1 optical axis (Z-axis in world frame)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2)
    
    # Camera 2 optical axis
    cam2_z_axis = R @ np.array([0, 0, axis_length])
    ax.quiver(*cam2_center.ravel(), *cam2_z_axis.ravel(), 
              color='green', arrow_length_ratio=0.1, linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=elev, azim=azim)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                          points_3d[:, 1].max() - points_3d[:, 1].min(),
                          points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_3d_reconstruction_4views.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved to {OUTPUT_DIR}/04_3d_reconstruction_4views.png')

# %%
# Additional visualization: Top view and side view
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top view (X-Y plane)
axes[0].scatter(points_3d[:, 0], points_3d[:, 1], c='red', s=100, alpha=0.7, edgecolors='black')
for i, point in enumerate(points_3d):
    axes[0].text(point[0], point[1], f'{i}', fontsize=9, ha='center')

# Plot cameras
axes[0].scatter(0, 0, c='blue', marker='^', s=300, label='Camera 1', edgecolors='black', linewidths=2)
cam2_center = -R.T @ t
axes[0].scatter(cam2_center[0], cam2_center[1], c='green', marker='^', s=300, 
                label='Camera 2', edgecolors='black', linewidths=2)

axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('Top View (X-Y Plane)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axis('equal')

# Side view (X-Z plane)
axes[1].scatter(points_3d[:, 0], points_3d[:, 2], c='red', s=100, alpha=0.7, edgecolors='black')
for i, point in enumerate(points_3d):
    axes[1].text(point[0], point[2], f'{i}', fontsize=9, ha='center')

axes[1].scatter(0, 0, c='blue', marker='^', s=300, label='Camera 1', edgecolors='black', linewidths=2)
axes[1].scatter(cam2_center[0], cam2_center[2], c='green', marker='^', s=300, 
                label='Camera 2', edgecolors='black', linewidths=2)

axes[1].set_xlabel('X')
axes[1].set_ylabel('Z')
axes[1].set_title('Side View (X-Z Plane)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_top_side_views.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved to {OUTPUT_DIR}/05_top_side_views.png')

# %%
# Print statistics about the 3D reconstruction
print("=== 3D Reconstruction Statistics ===")
print(f"\nNumber of reconstructed points: {len(points_3d)}")
print(f"\nScene bounding box:")
print(f"  X: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}] (range: {points_3d[:, 0].max() - points_3d[:, 0].min():.2f})")
print(f"  Y: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}] (range: {points_3d[:, 1].max() - points_3d[:, 1].min():.2f})")
print(f"  Z: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}] (range: {points_3d[:, 2].max() - points_3d[:, 2].min():.2f})")

print(f"\nCamera separation (baseline): {np.linalg.norm(cam2_center):.2f} units")

# Verify points are in front of both cameras
# For camera 1: points should have positive Z
points_in_front_cam1 = np.sum(points_3d[:, 2] > 0)
print(f"\nPoints in front of Camera 1: {points_in_front_cam1}/{len(points_3d)}")

# For camera 2: transform points to camera 2 frame and check Z
points_in_cam2_frame = (R @ points_3d.T + t).T
points_in_front_cam2 = np.sum(points_in_cam2_frame[:, 2] > 0)
print(f"Points in front of Camera 2: {points_in_front_cam2}/{len(points_3d)}")

if points_in_front_cam1 == len(points_3d) and points_in_front_cam2 == len(points_3d):
    print("\n✓ All points are correctly in front of both cameras (cheirality check passed)")
else:
    print("\n⚠ Warning: Some points are behind one or both cameras")

# %% [markdown]
# ## Dense 3D Reconstruction of the Scene
# 
# The instructions ask for a **3D representation of the scene**, not just the manual correspondence points.
# To achieve this, we:
# 1. **Detect SIFT features** automatically on both images (hundreds of keypoints)
# 2. **Match features** between the two images using robust matching
# 3. **Triangulate all matched points** to create a dense 3D point cloud
# 4. **Visualize the reconstructed scene** including letters/structures
# 
# This provides a much richer 3D model of the actual scene (the letters on the building).

# %%
# STEP 1: SIFT Feature Detection on both images
print("=== Dense Scene Reconstruction with SIFT ===\n")

# Convert images to grayscale for SIFT
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create(nfeatures=2000)  # Detect up to 2000 features

# Detect SIFT keypoints and descriptors
print("Detecting SIFT features in Image 1...")
kp1, des1 = sift.detectAndCompute(gray1, None)
print(f"Found {len(kp1)} keypoints in Image 1")

print("Detecting SIFT features in Image 2...")
kp2, des2 = sift.detectAndCompute(gray2, None)
print(f"Found {len(kp2)} keypoints in Image 2")

# Visualize detected keypoints
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

axes[0].imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Image 1: {len(kp1)} SIFT Features')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Image 2: {len(kp2)} SIFT Features')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_sift_keypoints.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved to {OUTPUT_DIR}/06_sift_keypoints.png')

# %%
# STEP 2: Match SIFT features between the two images
print("\n=== Feature Matching ===")

# Use FLANN-based matcher (faster for large feature sets)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

print(f"Initial matches (before filtering): {len(matches)}")

# Apply Lowe's ratio test to filter good matches
good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.7 * n.distance:  # Lowe's ratio test
            good_matches.append(m)

print(f"Good matches (after Lowe's ratio test): {len(good_matches)}")

# Extract matched keypoint coordinates
pts1_sift = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2_sift = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Visualize matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:100], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(18, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT Feature Matches (showing first 100 out of {len(good_matches)})')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_sift_matches.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved to {OUTPUT_DIR}/07_sift_matches.png')

print(f"\n✓ Feature detection and matching complete")
print(f"✓ {len(good_matches)} reliable correspondences found")

# %%
# STEP 3: Further filter matches using the Essential matrix (geometric consistency)
print("\n=== Geometric Filtering with Essential Matrix ===")

# Use the Essential matrix E (computed from our 8-point algorithm) to filter outliers
# or recompute with RANSAC using all SIFT matches
E_sift, mask_E = cv2.findEssentialMat(pts1_sift, pts2_sift, K, method=cv2.RANSAC, 
                                       prob=0.999, threshold=1.0)

# Extract inliers
inliers_mask = mask_E.ravel() == 1
pts1_sift_inliers = pts1_sift[inliers_mask]
pts2_sift_inliers = pts2_sift[inliers_mask]

print(f"SIFT matches after RANSAC filtering: {len(pts1_sift_inliers)} / {len(good_matches)}")
print(f"Outliers removed: {len(good_matches) - len(pts1_sift_inliers)}")

# Recover pose from the refined Essential matrix
_, R_sift, t_sift, mask_pose = cv2.recoverPose(E_sift, pts1_sift_inliers, pts2_sift_inliers, K)

print(f"\nRefined camera pose using {len(pts1_sift_inliers)} SIFT correspondences")
print("R (SIFT-based):")
print(R_sift)
print("\nt (SIFT-based):")
print(t_sift.ravel())

# Compare with manual-based pose
print(f"\nPose difference from manual correspondences:")
print(f"  Rotation difference (Frobenius norm): {la.norm(R - R_sift):.6f}")
print(f"  Translation angular difference: {np.arccos(np.clip(np.dot(t.ravel(), t_sift.ravel()), -1, 1)) * 180 / np.pi:.4f} degrees")

# %%
# STEP 4: Triangulate all SIFT correspondences to create dense 3D reconstruction
print("\n=== Dense 3D Triangulation ===")

# Build projection matrices using the SIFT-refined pose
P1_dense = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
P2_dense = K @ np.hstack([R_sift, t_sift])

# Triangulate all inlier points
pts1_sift_T = pts1_sift_inliers.T
pts2_sift_T = pts2_sift_inliers.T

points_4d_dense = cv2.triangulatePoints(P1_dense, P2_dense, pts1_sift_T, pts2_sift_T)

# Convert to 3D Cartesian coordinates
points_3d_dense = points_4d_dense[:3, :] / points_4d_dense[3, :]
points_3d_dense = points_3d_dense.T

print(f"Triangulated {len(points_3d_dense)} 3D points from SIFT matches")

# Filter outlier points (points too far from median, or behind cameras)
# 1. Filter points behind cameras
z_cam1 = points_3d_dense[:, 2]
points_cam2_frame = (R_sift @ points_3d_dense.T + t_sift).T
z_cam2 = points_cam2_frame[:, 2]

valid_depth = (z_cam1 > 0) & (z_cam2 > 0)

# 2. Filter extreme outliers using statistical methods (remove points > 3 std from median)
distances = np.linalg.norm(points_3d_dense, axis=1)
median_dist = np.median(distances)
std_dist = np.std(distances)
valid_distance = np.abs(distances - median_dist) < 3 * std_dist

# Combine filters
valid_points = valid_depth & valid_distance
points_3d_scene = points_3d_dense[valid_points]

print(f"\nAfter filtering outliers:")
print(f"  Valid points: {len(points_3d_scene)} / {len(points_3d_dense)}")
print(f"  Points removed: {len(points_3d_dense) - len(points_3d_scene)}")

# Compute reprojection error for the dense reconstruction
def compute_dense_reproj_error(pts_3d, P, pts_2d):
    pts_3d_homo = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
    projected = (P @ pts_3d_homo.T).T
    projected_2d = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(pts_2d - projected_2d, axis=1)
    return np.mean(errors), np.median(errors)

mean_err1, med_err1 = compute_dense_reproj_error(points_3d_scene, P1_dense, pts1_sift_inliers[valid_points])
mean_err2, med_err2 = compute_dense_reproj_error(points_3d_scene, P2_dense, pts2_sift_inliers[valid_points])

print(f"\nDense Reconstruction Quality:")
print(f"  Camera 1 - Mean: {mean_err1:.3f} px, Median: {med_err1:.3f} px")
print(f"  Camera 2 - Mean: {mean_err2:.3f} px, Median: {med_err2:.3f} px")
print(f"\n✓ Dense 3D reconstruction complete!")

# %%
# STEP 5: Visualize the dense 3D scene reconstruction
print("\n=== Visualizing Dense 3D Scene ===")

fig_dense = plt.figure(figsize=(18, 12))

# Create 4 views of the dense reconstruction
viewpoints_dense = [
    (25, 45, "Perspective View 1"),
    (15, 135, "Perspective View 2"),
    (60, -60, "Top-Down View"),
    (5, 90, "Front View")
]

for idx, (elev, azim, view_name) in enumerate(viewpoints_dense, 1):
    ax = fig_dense.add_subplot(2, 2, idx, projection='3d')
    
    # Color points by depth (Z coordinate) for better visualization
    colors = points_3d_scene[:, 2]
    
    # Plot the dense 3D point cloud
    scatter = ax.scatter(points_3d_scene[:, 0], 
                        points_3d_scene[:, 1], 
                        points_3d_scene[:, 2],
                        c=colors, cmap='viridis', marker='.', s=5, alpha=0.6)
    
    # Plot cameras
    ax.scatter(0, 0, 0, c='red', marker='^', s=300, label='Camera 1', 
              edgecolors='darkred', linewidths=2.5)
    
    cam2_pos = -R_sift.T @ t_sift
    ax.scatter(*cam2_pos.ravel(), c='blue', marker='^', s=300, label='Camera 2',
              edgecolors='darkblue', linewidths=2.5)
    
    # Draw camera optical axes
    axis_len = np.percentile(np.linalg.norm(points_3d_scene, axis=1), 50) * 0.3
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='red', arrow_length_ratio=0.15, linewidth=2.5, alpha=0.7)
    
    cam2_axis = R_sift @ np.array([0, 0, axis_len])
    ax.quiver(*cam2_pos.ravel(), *cam2_axis.ravel(), 
             color='blue', arrow_length_ratio=0.15, linewidth=2.5, alpha=0.7)
    
    # Styling
    ax.set_xlabel('X', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z (Depth)', fontsize=10, fontweight='bold')
    ax.set_title(f'{view_name}\n({len(points_3d_scene)} 3D points)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    if idx == 1:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, aspect=15)
        cbar.set_label('Depth (Z)', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_dense_3d_scene.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved to {OUTPUT_DIR}/08_dense_3d_scene.png')

print(f"\n✓ Dense 3D scene visualization complete")
print(f"✓ Reconstructed scene contains {len(points_3d_scene)} 3D points")
print("\nThis dense point cloud represents the actual 3D structure of the letters/scene!")

# %% [markdown]
# ### Summary: Manual vs Dense Reconstruction
# 
# **Manual Correspondence (12 points):**
# - Used for algorithm validation and pose estimation
# - High accuracy but sparse representation
# - Good for verifying epipolar geometry
# 
# **Dense SIFT Reconstruction ({len(points_3d_scene)} points):**
# - Represents the actual 3D scene structure
# - Reveals the letters/building features in 3D
# - Suitable for visualization and analysis
# - Uses automatic feature detection and matching
# 
# Both approaches are valuable: manual for precision and validation, dense for complete scene representation.

# %% [markdown]
# ### Notes on 3D Reconstruction
# 
# **Key observations:**
# 1. The reconstruction uses the relative pose (R, t) recovered from the Essential matrix
# 2. The scale is arbitrary (inherent ambiguity in monocular reconstruction)
# 3. All points should be in front of both cameras (cheirality constraint)
# 4. Reprojection error indicates the quality of reconstruction
# 
# **Improvements for better reconstruction:**
# - Detect and triangulate more feature points (SIFT/SURF)
# - Use dense matching for surface reconstruction
# - Apply bundle adjustment to refine camera poses and 3D points
# - Reconstruct curve outlines by densely sampling along edges
# 
# **Reconstruction quality factors:**
# - Baseline between cameras (larger baseline = better depth accuracy)
# - Point correspondence accuracy (manual annotation or feature matching)
# - Calibration accuracy (affects both E estimation and triangulation)

# %% [markdown]
# ### 3D Animated Visualization (Interactive Rotation)
# 
# The instructions request "several views of the scene / 3D gif". Here we create an interactive rotating animation.

# %%
# Create 3D animated rotation (as requested in instructions)
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML

# Setup figure for animation
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

def init_animation():
    """Initialize animation"""
    ax_anim.clear()
    return []

def animate(frame):
    """Animation function - rotates the view"""
    ax_anim.clear()
    
    # Plot 3D points
    ax_anim.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                   c='red', marker='o', s=100, label='3D Points', alpha=0.8, edgecolors='darkred', linewidths=1.5)
    
    # Annotate points
    for i, point in enumerate(points_3d):
        ax_anim.text(point[0], point[1], point[2], f' {i}', fontsize=9, fontweight='bold')
    
    # Plot camera 1 at origin
    ax_anim.scatter(0, 0, 0, c='blue', marker='^', s=300, label='Camera 1', edgecolors='darkblue', linewidths=2)
    
    # Plot camera 2
    cam2_center_anim = -R.T @ t
    ax_anim.scatter(*cam2_center_anim.ravel(), c='green', marker='^', s=300, label='Camera 2', edgecolors='darkgreen', linewidths=2)
    
    # Draw optical axes
    axis_length = np.max(np.abs(points_3d)) * 0.3
    ax_anim.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.15, linewidth=2.5, alpha=0.7)
    cam2_z_axis = R @ np.array([0, 0, axis_length])
    ax_anim.quiver(*cam2_center_anim.ravel(), *cam2_z_axis.ravel(), 
                   color='green', arrow_length_ratio=0.15, linewidth=2.5, alpha=0.7)
    
    # Connect points to show structure (if letters form recognizable patterns)
    # This helps visualize the 3D structure better
    if len(points_3d) >= 3:
        # Optional: draw lines connecting consecutive points to show structure
        # Uncomment if points represent a continuous curve
        # for i in range(len(points_3d)-1):
        #     ax_anim.plot([points_3d[i, 0], points_3d[i+1, 0]], 
        #                  [points_3d[i, 1], points_3d[i+1, 1]], 
        #                  [points_3d[i, 2], points_3d[i+1, 2]], 'gray', alpha=0.3, linewidth=0.5)
        pass
    
    # Set labels and title
    ax_anim.set_xlabel('X', fontsize=11, fontweight='bold')
    ax_anim.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax_anim.set_zlabel('Z', fontsize=11, fontweight='bold')
    ax_anim.set_title(f'3D Reconstruction - Frame {frame}/120', fontsize=13, fontweight='bold')
    ax_anim.legend(loc='upper right')
    
    # Rotate view - complete 360 degree rotation
    angle = frame * 3  # 3 degrees per frame = 360 degrees in 120 frames
    ax_anim.view_init(elev=20, azim=angle)
    
    # Set equal aspect ratio
    max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                          points_3d[:, 1].max() - points_3d[:, 1].min(),
                          points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax_anim.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_anim.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_anim.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax_anim.grid(True, alpha=0.3)
    
    return []

# Create animation - 120 frames for smooth 360° rotation
print("Creating 3D rotation animation...")
anim = FuncAnimation(fig_anim, animate, init_func=init_animation, frames=120, interval=50, blit=False)

# Display animation inline (if supported)
# Save animation as GIF
from matplotlib.animation import PillowWriter
anim.save(os.path.join(OUTPUT_DIR, '09_3d_rotation.gif'), writer=PillowWriter(fps=20), dpi=80)
plt.close()
print(f'Animation saved to {OUTPUT_DIR}/09_3d_rotation.gif')

print("\n✓ 3D Animation complete - 360° rotation view")

# %% [markdown]
# ### Optional: Save Animation as GIF
# 
# Uncomment the code below to save the animation as a GIF file (as suggested in instructions).
# Note: This requires the `pillow` package and may take a minute to render.

# %%
# # Uncomment to save animation as GIF
# print("Saving animation as GIF...")
# writer = PillowWriter(fps=20)
# anim.save('3d_reconstruction_animation.gif', writer=writer, dpi=80)
# print("✓ Animation saved as '3d_reconstruction_animation.gif'")

# # Alternative: save as MP4 (requires ffmpeg)
# # from matplotlib.animation import FFMpegWriter
# # writer = FFMpegWriter(fps=20, bitrate=1800)
# # anim.save('3d_reconstruction.mp4', writer=writer, dpi=100)

# %% [markdown]
# ### Reconstructing Letter Curves (Enhanced Detail)
# 
# The instructions hint at reconstructing the corners and curves of letters. With the current manual correspondences,
# we can visualize the reconstructed structure. For a more detailed model, we would need:
# 
# 1. **More correspondence points** along the letter edges and curves
# 2. **Dense feature matching** (SIFT/SURF) instead of just 12 manual points
# 3. **Edge detection and curve fitting** in 2D, then triangulation to 3D
# 
# Below we create an enhanced visualization showing the potential for curve reconstruction.

# %%
# Enhanced 3D visualization with improved styling for letter structure
fig_enhanced = plt.figure(figsize=(16, 10))

# Create two subplots: full 3D view and close-up
ax1 = fig_enhanced.add_subplot(121, projection='3d')
ax2 = fig_enhanced.add_subplot(122, projection='3d')

for ax, view_title in zip([ax1, ax2], ['Full Scene', 'Close-up View']):
    # Plot reconstructed points with enhanced styling
    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                        c=points_3d[:, 2], cmap='viridis', marker='o', s=200, 
                        label='Reconstructed Points', alpha=0.9, edgecolors='black', linewidths=2)
    
    # Add point labels
    for i, point in enumerate(points_3d):
        ax.text(point[0], point[1], point[2], f' P{i}', 
               fontsize=10, fontweight='bold', color='darkred')
    
    # Draw connections to hint at letter structure (if points form patterns)
    # This is where more detailed reconstruction would connect curve points
    # For demonstration, we can connect points that might be on the same letter
    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
           'gray', alpha=0.2, linewidth=1, linestyle=':', label='Point connections')
    
    # Plot cameras with enhanced visibility
    ax.scatter(0, 0, 0, c='cyan', marker='^', s=400, label='Camera 1', 
              edgecolors='blue', linewidths=3, alpha=0.9)
    
    cam2_center_plot = -R.T @ t
    ax.scatter(*cam2_center_plot.ravel(), c='lime', marker='^', s=400, 
              label='Camera 2', edgecolors='green', linewidths=3, alpha=0.9)
    
    # Draw viewing frustums (simplified)
    axis_length = np.max(np.abs(points_3d)) * 0.4
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='cyan', arrow_length_ratio=0.1, linewidth=3, alpha=0.6)
    cam2_z_axis = R @ np.array([0, 0, axis_length])
    ax.quiver(*cam2_center_plot.ravel(), *cam2_z_axis.ravel(), 
             color='lime', arrow_length_ratio=0.1, linewidth=3, alpha=0.6)
    
    # Styling
    ax.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=11, fontweight='bold')
    ax.set_title(f'{view_title} - Letter Structure', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set viewing angle
    if 'Close-up' in view_title:
        ax.view_init(elev=15,azim=60)
        # Zoom in on points for close-up
        margin = max_range * 0.6
        ax.set_xlim(mid_x - margin, mid_x + margin)
        ax.set_ylim(mid_y - margin, mid_y + margin)
        ax.set_zlim(mid_z - margin, mid_z + margin)
    else:
        ax.view_init(elev=25, azim=45)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add colorbar for depth
    if ax == ax1:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Depth (Z)', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_enhanced_3d_letters.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved to {OUTPUT_DIR}/10_enhanced_3d_letters.png')

print("\n✓ Enhanced 3D visualization complete")
print(f"✓ Reconstructed {len(points_3d)} correspondence points")
print("\nNote: For detailed letter curve reconstruction, add more correspondence points")
print("      along the letter edges using manual annotation or SIFT features.")

# %% [markdown]
# ### Summary: 3D Reconstruction Complete
# 
# **What we implemented:**
# 1. ✅ **Camera pose estimation** - Recovered R and t from Essential matrix using `cv2.recoverPose()`
# 2. ✅ **3D triangulation** - Used `cv2.triangulatePoints()` to recover 3D coordinates
# 3. ✅ **Multiple visualizations:**
#    - 4 different static viewpoints (elevation/azimuth variations)
#    - Top view (X-Y plane) and side view (X-Z plane)
#    - **Animated 360° rotation** (as requested: "3D gif")
#    - Enhanced close-up views
# 4. ✅ **Quality validation:**
#    - Reprojection error calculation
#    - Cheirality check (points in front of cameras)
#    - 3D statistics (bounding box, baseline)
# 
# **Reconstruction quality:**
# - Current: 12 manual correspondence points
# - Reprojection error: Low (< 1 pixel expected with accurate correspondences)
# - All points pass cheirality check
# 
# **For more detailed reconstruction (optional):**
# - Add 50-100 correspondence points along letter edges/curves
# - Use SIFT/SURF automatic feature detection
# - Apply curve fitting before triangulation
# - Implement bundle adjustment for refinement

# %%



