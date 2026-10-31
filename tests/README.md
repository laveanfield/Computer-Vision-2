# Testing Guide & Protocols: Computer Vision 2

## 1. Test Suite Architecture & Organization
The test suite is located under the `tests/` directory and is strictly divided into **Unit Tests** (`unit_test/`) and **End-to-End Tests** (`e2e_test/`) for each subsystem.

```text
tests/
├── image_blending/                       # Part 1: Multi-resolution & Poisson Blending
│   ├── e2e_test/
│   │   └── test_blending_pipeline_e2e.py # End-to-end multi-image blending integration
│   └── unit_test/
│       ├── test_config.py                # Hyperparameter schemas & validation
│       ├── test_laplacian_blend.py       # Gaussian & Laplacian pyramid blending
│       ├── test_naive_blend.py           # Cut-and-paste linear alpha baseline
│       ├── test_poisson_blend.py         # Discrete Poisson PDE solver (gradient domain)
│       └── test_utils.py                 # Color conversions, gradients & pyramid math
└── panorama/                             # Part 2: Feature-based Panorama Stitching
    ├── e2e_test/
    │   └── test_stitcher_e2e.py          # Complete multi-image stitching pipeline
    └── unit_test/
        ├── test_blend.py                 # Multiband & feathering seam blending
        ├── test_config.py                # Stitching configurations & dataclasses
        ├── test_features.py              # SIFT/ORB feature extractors & keypoints
        ├── test_harris_mops.py           # Harris corners, ANMS & MOPS descriptors
        ├── test_homography.py            # DLT, conditioning, RANSAC & LM refinement
        ├── test_matching.py              # Nearest-neighbor matching & Lowe ratio test
        ├── test_metrics.py               # PSNR, SSIM & reprojection metrics
        ├── test_synthetic.py             # Geometric warp generators & sweeps
        └── test_warp.py                  # Projective transformation & canvas layout
```

### Test Count & Coverage Overview
The repository enforces **100.00% statement and branch test coverage** across all production modules:
- **Image Blending**: 6 test suites, **40 tests** (100% statement & branch coverage).
- **Panorama**: 10 test suites, **100 tests** (100% statement & branch coverage).
- **Total**: 16 test suites, **140 tests** ($\approx 9.3\text{s}$ execution time).

---

## 2. Test Execution Commands
All tests use `pytest` and `pytest-cov`.

### 2.1 Full Test Suite Execution
Run all 140 tests across both subsystems with line and branch coverage reporting, plus HTML dashboard generation:
```bash
python -m pytest --cov=src --cov-report=term-missing --cov-report=html:coverage_html tests/
```

### 2.2 Subsystem-Targeted Execution
To run only the **Image Blending** tests:
```bash
python -m pytest --cov=src/image_blending --cov-report=term-missing tests/image_blending/
```

To run only the **Panorama** tests:
```bash
python -m pytest --cov=src/panorama --cov-report=term-missing tests/panorama/
```

### 2.3 Layer-Targeted Execution
To run only unit tests:
```bash
python -m pytest tests/image_blending/unit_test/ tests/panorama/unit_test/
```

To run only end-to-end integration tests:
```bash
python -m pytest tests/image_blending/e2e_test/ tests/panorama/e2e_test/
```

To run a specific test file:
```bash
python -m pytest tests/panorama/unit_test/test_homography.py -v
```

### 2.4 Viewing the HTML Coverage Dashboard
The HTML coverage report is generated in the `coverage_html/` directory. Open `coverage_html/index.html` in your default browser:

```bash
start coverage_html/index.html
```