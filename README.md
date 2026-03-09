# Computer Vision — Assignment 1

**Università Commerciale Luigi Bocconi** · Computer Vision and Image Processing

**Authors:** Gregorio Ceria, Tommaso Aiello, Alessandro Bottardi

---

## Overview

This project implements a full two-view geometry pipeline:

1. **Camera Calibration** — Intrinsic parameter estimation from checkerboard images using Zhang's method (OpenCV).
2. **Two-View Geometry** — Essential matrix estimation via the 8-point algorithm, SIFT feature matching, epipolar geometry verification, and triangulation.

## Repository Structure

| File / Folder | Description |
|---|---|
| `ex1hw.ipynb` | Exercise 1 — Camera calibration notebook |
| `ex2hwmanual.ipynb` | Exercise 2 — Two-view geometry notebook |
| `calibration.json` | Calibration output (intrinsic matrix & distortion coefficients) |
| `Photos2/` | Input images for stereo pair |
| `Report_CV1.pdf` | Compiled report with full methodology and results |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebooks in Jupyter or VS Code and run all cells sequentially.

## Dependencies

- Python 3.10+
- NumPy, OpenCV, Matplotlib, SciPy, Plotly, Pillow

## License

See [LICENSE](LICENSE).