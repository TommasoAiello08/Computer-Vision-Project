#!/usr/bin/env python3
"""
Ex.1 - Camera calibration from checkerboard images.

What this script does:
1) Loads calibration images from known folders.
2) Detects checkerboard corners (with automatic board-size diagnosis if needed).
3) Calibrates camera intrinsics with OpenCV.
4) Saves results to calibration.npz / calibration.json / calibration.yml.
5) Writes a detailed diagnostic report to output/calibration_diagnostics.json.

Run:
    python ex1hw.py
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np


# -------------------------- Configuration --------------------------
# If you know your board, set it here. (inner corners: cols, rows)
# Leave as None to auto-detect from CANDIDATE_PATTERNS.
KNOWN_CHECKERBOARD: Optional[Tuple[int, int]] = None

# Candidate inner-corner patterns to test if KNOWN_CHECKERBOARD is None.
# Includes common boards and square boards around your expected 14x14.
CANDIDATE_PATTERNS: List[Tuple[int, int]] = [
    (14, 14),
    (13, 13),
    (12, 12),
    (11, 11),
    (10, 10),
    (9, 9),
    (8, 8),
    (7, 7),
    (10, 7),
    (9, 6),
    (8, 6),
    (7, 6),
    (11, 8),
    (10, 8),
]

IMAGE_FOLDERS = ["Old_photos", "Photos2", "."]
IMAGE_EXTS = ("*.jpeg", "*.jpg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
MIN_VALID_VIEWS = 3
DIAG_MAX_IMAGES = 8

# Resize for faster pattern diagnosis only (full-res is kept for final detection)
DIAG_MAX_SIDE = 1800
DET_MAX_SIDE = 2000

# OpenCV corner refinement criteria
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50,
    1e-4,
)


@dataclass
class DetectionResult:
    success: bool
    corners: Optional[np.ndarray]
    method: str


def collect_images() -> List[Path]:
    files: List[Path] = []
    for folder in IMAGE_FOLDERS:
        for ext in IMAGE_EXTS:
            files.extend(Path(p) for p in glob.glob(str(Path(folder) / ext)))
    files = sorted(set(files))
    return files


def read_gray(image_path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return img


def preprocess_variants(gray: np.ndarray) -> List[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = cv2.equalizeHist(gray)
    clahe_img = clahe.apply(gray)
    return [gray, eq, clahe_img]


def detect_corners(gray: np.ndarray, pattern: Tuple[int, int]) -> DetectionResult:
    flags_classic = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    flags_sb = cv2.CALIB_CB_NORMALIZE_IMAGE

    h, w = gray.shape[:2]
    scale = min(1.0, DET_MAX_SIDE / max(h, w))
    if scale < 1.0:
        work = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        work = gray

    for variant in preprocess_variants(work):
        ok, corners = cv2.findChessboardCorners(variant, pattern, flags_classic)
        if ok and corners is not None:
            corners_refined = cv2.cornerSubPix(
                variant,
                corners,
                (11, 11),
                (-1, -1),
                SUBPIX_CRITERIA,
            )
            if scale < 1.0:
                corners_refined = corners_refined / scale
            return DetectionResult(True, corners_refined, "findChessboardCorners")

        try:
            ok_sb, corners_sb = cv2.findChessboardCornersSB(variant, pattern, flags_sb)
        except Exception:
            ok_sb, corners_sb = False, None
        if ok_sb and corners_sb is not None:
            if scale < 1.0:
                corners_sb = corners_sb / scale
            return DetectionResult(True, corners_sb, "findChessboardCornersSB")

    return DetectionResult(False, None, "none")


def resized_for_diag(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = DIAG_MAX_SIDE / max(h, w)
    if scale < 1.0:
        return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray


def diagnose_best_pattern(images: List[Path], patterns: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Dict[str, int]]:
    pattern_hits: Dict[str, int] = {f"{p[0]}x{p[1]}": 0 for p in patterns}

    # Fast diagnosis: use only a subset to avoid long runtime.
    diag_images = images[: min(len(images), DIAG_MAX_IMAGES)]

    for image_path in diag_images:
        gray = read_gray(image_path)
        if gray is None:
            continue
        gray_diag = resized_for_diag(gray)
        for pattern in patterns:
            ok, _ = cv2.findChessboardCorners(
                gray_diag,
                pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ok:
                pattern_hits[f"{pattern[0]}x{pattern[1]}"] += 1

    best_pattern = patterns[0]
    best_count = -1
    for pattern in patterns:
        count = pattern_hits[f"{pattern[0]}x{pattern[1]}"]
        if count > best_count:
            best_count = count
            best_pattern = pattern

    return best_pattern, pattern_hits


def make_object_points(pattern: Tuple[int, int]) -> np.ndarray:
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    return objp


def ensure_output_dirs() -> None:
    Path("output").mkdir(exist_ok=True)
    Path("output/calib_debug").mkdir(parents=True, exist_ok=True)


def main() -> int:
    cv2.ocl.setUseOpenCL(False)
    ensure_output_dirs()

    images = collect_images()
    print(f"Found {len(images)} candidate images from folders: {IMAGE_FOLDERS}")

    if not images:
        print("No images found. Calibration aborted.")
        return 1

    if KNOWN_CHECKERBOARD is None:
        checkerboard, pattern_hits = diagnose_best_pattern(images, CANDIDATE_PATTERNS)
    else:
        checkerboard = KNOWN_CHECKERBOARD
        pattern_hits = {f"{KNOWN_CHECKERBOARD[0]}x{KNOWN_CHECKERBOARD[1]}": 0}

    print(f"Using checkerboard inner corners: {checkerboard}")

    objp = make_object_points(checkerboard)
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    used_images: List[str] = []
    rejected_images: List[str] = []
    methods_used: Dict[str, int] = {}
    image_size: Optional[Tuple[int, int]] = None

    for image_path in images:
        gray = read_gray(image_path)
        if gray is None:
            rejected_images.append(str(image_path))
            continue

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        result = detect_corners(gray, checkerboard)

        if result.success and result.corners is not None:
            objpoints.append(objp.copy())
            imgpoints.append(result.corners)
            used_images.append(str(image_path))
            methods_used[result.method] = methods_used.get(result.method, 0) + 1

            color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            dbg = cv2.drawChessboardCorners(color.copy(), checkerboard, result.corners, True)
            out_path = Path("output/calib_debug") / f"corners_{image_path.name}"
            cv2.imwrite(str(out_path), dbg)
            print(f"[OK] corners detected in {image_path.name} via {result.method}")
        else:
            rejected_images.append(str(image_path))
            print(f"[NO] corners not detected in {image_path.name}")

    diagnostics = {
        "image_folders": IMAGE_FOLDERS,
        "num_candidate_images": len(images),
        "chosen_checkerboard": list(checkerboard),
        "pattern_hits": pattern_hits,
        "num_valid_views": len(used_images),
        "used_images": used_images,
        "rejected_images": rejected_images,
        "methods_used": methods_used,
    }

    if len(objpoints) < MIN_VALID_VIEWS or image_size is None:
        diagnostics["status"] = "failed"
        diagnostics["reason"] = (
            f"Not enough valid checkerboard views: {len(objpoints)} found, need at least {MIN_VALID_VIEWS}."
        )
        with open("output/calibration_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)
        print(diagnostics["reason"])
        print("Wrote diagnostics to output/calibration_diagnostics.json")
        return 1

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    per_view_errors: List[float] = []
    for i in range(len(objpoints)):
        reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], reprojected, cv2.NORM_L2) / len(reprojected)
        per_view_errors.append(float(err))

    mean_reproj_error = float(np.mean(per_view_errors))

    np.savez(
        "calibration.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        reprojection_error=mean_reproj_error,
        rms=rms,
        checkerboard=np.array(checkerboard, dtype=np.int32),
    )

    calibration_json = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "reprojection_error": mean_reproj_error,
        "rms": float(rms),
        "checkerboard": list(checkerboard),
    }
    with open("calibration.json", "w") as f:
        json.dump(calibration_json, f, indent=2)

    fs = cv2.FileStorage("calibration.yml", cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("reprojection_error", mean_reproj_error)
    fs.write("rms", float(rms))
    fs.write("checkerboard", np.array(checkerboard, dtype=np.int32))
    fs.release()

    diagnostics.update(
        {
            "status": "success",
            "rms": float(rms),
            "reprojection_error": mean_reproj_error,
            "per_view_reprojection_error": per_view_errors,
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.ravel().tolist(),
        }
    )

    with open("output/calibration_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("\n=== Calibration successful ===")
    print("camera_matrix:\n", camera_matrix)
    print("dist_coeffs:", dist_coeffs.ravel())
    print(f"rms: {float(rms):.6f}")
    print(f"mean reprojection error: {mean_reproj_error:.6f} px")
    print("Saved calibration.npz, calibration.json, calibration.yml")
    print("Wrote diagnostics to output/calibration_diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
