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
    plt.show()

else:
    # -----------------------------
    # FULL COMPUTE (più lento)
    # -----------------------------
    # Caricamento calibrazione
    with open('calibration.json', 'r') as f:
        calib = json.load(f)

    K = np.array(calib['camera_matrix'], dtype=np.float64)
    dist = np.array(calib['dist_coeffs'], dtype=np.float64).reshape(-1, 1)

    # Caricamento immagini
    img1 = cv2.imread('Photos2/img1.jpeg')
    img2 = cv2.imread('Photos2/img2.jpeg')
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
    plt.show()

    np.savez(SAVED_PATH, points1=points1, points2=points2, F=F)