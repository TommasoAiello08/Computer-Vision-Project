import numpy as np

# Your originals
immagine_storta = np.array([
    (1531, 1654), (1878, 1651), (2203, 1622), (2305, 1629), (2349, 1973),
    (2593, 1634), (2600, 1973), (2726, 1716), (2872, 1979), (2848, 1851),
    (2968, 1856), (3295, 1884), (3425, 1846), (3558, 1716), (3456, 1979),
    (3725, 1979), (3934, 1721), (3991, 1981), (3885, 1702), (4093, 1657),
    (3158, 1168), (4659, 2241)
], dtype=np.float32)

immagine_dritta = np.array([
    (1284, 1675), (1515, 1687), (1983, 1669), (2281, 1676), (2306, 2052),
    (2289, 1695), (2298, 2058), (2633, 1778), (2652, 2040), (2843, 1916),
    (2832, 1918), (3247, 1958), (3193, 1912), (3444, 1796), (3448, 2053),
    (3618, 2053), (3657, 1844), (3731, 2052), (3948, 1786), (3942, 1749),
    (2268, 1245), (3891, 2287)
], dtype=np.float32)

def densify_correspondences(pts2, pts1, n_between=2, max_dist=700):
    """
    pts2: points in 'immagine_storta'
    pts1: corresponding points in 'immagine_dritta'
    n_between: how many new points to insert between consecutive pairs
    max_dist: skip inserting if the segment is too long (likely a jump between features)
    """
    A, B = [], []
    for i in range(len(pts2) - 1):
        p2, q2 = pts2[i], pts2[i+1]
        p1, q1 = pts1[i], pts1[i+1]

        A.append(p2); B.append(p1)

        d2 = np.linalg.norm(q2 - p2)
        d1 = np.linalg.norm(q1 - p1)
        if (d2 + d1) / 2.0 <= max_dist:
            for t in range(1, n_between + 1):
                alpha = t / (n_between + 1)
                A.append(p2 * (1 - alpha) + q2 * alpha)
                B.append(p1 * (1 - alpha) + q1 * alpha)

    A.append(pts2[-1]); B.append(pts1[-1])
    return np.array(A, dtype=np.float32), np.array(B, dtype=np.float32)

# Example: +2 points between each “local” pair
immagine_storta_aug, immagine_dritta_aug = densify_correspondences(
    immagine_storta, immagine_dritta,
    n_between=2,
    max_dist=700
)

print("Original:", len(immagine_storta), "Augmented:", len(immagine_storta_aug))