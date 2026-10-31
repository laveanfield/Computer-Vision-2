"""
Shared pytest fixtures and test helpers for panorama tests.
Implements the Input-Output model for reusable test data setup.
"""

from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest

from panorama.config import PANO_INPUT_DIR


@pytest.fixture
def synthetic_corner_image() -> np.ndarray:
    """
    [INPUT]: None.
    [OUTPUT]: 120x120 uint8 grayscale image containing high-contrast squares and textured checker patterns
              guaranteed to fire Harris, SIFT, and ORB corner detectors.
    """
    img = np.full((120, 120), 128, dtype=np.uint8)
    # Draw high-contrast shapes with distinct corners
    cv2.rectangle(img, (20, 20), (50, 50), 255, -1)
    cv2.rectangle(img, (70, 20), (100, 50), 0, -1)
    cv2.circle(img, (60, 85), 20, 240, -1)
    cv2.rectangle(img, (25, 75), (45, 105), 30, -1)
    return img


@pytest.fixture
def synthetic_homography_pair(synthetic_corner_image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [INPUT]: synthetic_corner_image.
    [OUTPUT]:
        tuple of (img1, img2, H_true):
        - img1: Original synthetic image (120x120 uint8).
        - img2: Image warped by a known Euclidean similarity transformation (rotation 5 deg + translation [10, 5]).
        - H_true: The exact 3x3 ground-truth homography matrix mapping img1 -> img2.
    """
    img1 = synthetic_corner_image
    angle_rad = np.deg2rad(5.0)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    tx, ty = 8.0, 4.0

    H_true = np.array([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)

    img2 = cv2.warpPerspective(img1, H_true, (120, 120), flags=cv2.INTER_LINEAR)
    return img1, img2, H_true


@pytest.fixture
def sample_mountain_paths() -> list[Path]:
    """
    [INPUT]: None.
    [OUTPUT]: list of Path objects for the mountain scene (mountain1.png, mountain2.png).
    """
    mountain_dir = PANO_INPUT_DIR / "mountain"
    if not mountain_dir.exists():
        pytest.skip("Mountain dataset not found.")
    paths = sorted(mountain_dir.glob("*.png"))
    if len(paths) < 2:
        pytest.skip("Insufficient images in mountain dataset.")
    return paths


@pytest.fixture
def sample_aqueduct_paths() -> list[Path]:
    """
    [INPUT]: None.
    [OUTPUT]: list of Path objects for the aqueduct scene (aqueduct1.jpg, aqueduct2.jpg).
    """
    aqueduct_dir = PANO_INPUT_DIR / "aqueduct"
    if not aqueduct_dir.exists():
        pytest.skip("Aqueduct dataset not found.")
    paths = sorted(aqueduct_dir.glob("*.jpg"))
    if len(paths) < 2:
        pytest.skip("Insufficient images in aqueduct dataset.")
    return paths
