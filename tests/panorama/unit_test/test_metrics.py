"""
Unit tests for panorama.metrics module following the Input-Output testing model.
"""

import numpy as np
import pytest
from panorama.metrics import (
    _luminance,
    _ncc,
    overlap_photometric_error,
    overlap_ncc_bands,
    homography_rms_error,
)


def test_luminance_conversion():
    """
    [INPUT]: Uniform pure red BGR image [0.0, 0.0, 1.0].
    [OUTPUT]: Luminance equals 0.299 (standard BT.601 red coefficient).
    """
    # --- [1. INPUT] ---
    img_red_bgr = np.zeros((10, 10, 3), dtype=np.float32)
    img_red_bgr[:, :, 2] = 1.0  # Red channel in BGR

    # --- [2. ACT] ---
    lum = _luminance(img_red_bgr)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(lum, 0.299, atol=1e-5)


def test_ncc_identity_and_inversion():
    """
    [INPUT]:
        - Signal a.
        - Signal b identical to a.
        - Signal c negatively correlated (c = -a).
    [OUTPUT]:
        - _ncc(a, b) == 1.0.
        - _ncc(a, c) == -1.0.
    """
    # --- [1. INPUT] ---
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a.copy()
    c = -a

    # --- [2. ACT] ---
    ncc_same = _ncc(a, b)
    ncc_inv = _ncc(a, c)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert np.isclose(ncc_same, 1.0)
    assert np.isclose(ncc_inv, -1.0)


def test_overlap_photometric_error_identical_images():
    """
    [INPUT]: Two identical warped 40x40 images with full overlap.
    [OUTPUT]: Overlap error metrics show mae == 0.0 and ncc == 1.0.
    """
    # --- [1. INPUT] ---
    np.random.seed(42)
    img = np.random.uniform(0.2, 0.8, size=(40, 40, 3)).astype(np.float32)
    mask = np.ones((40, 40), dtype=bool)

    # --- [2. ACT] ---
    metrics = overlap_photometric_error(img, mask, img, mask, erode=2, min_pixels=100)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert metrics["n_pixels"] > 0
    assert np.isclose(metrics["mae"], 0.0, atol=1e-5)
    assert np.isclose(metrics["ncc"], 1.0, atol=1e-4)


def test_overlap_photometric_error_insufficient_overlap():
    """
    [INPUT]: Two images with non-overlapping masks (0 overlapping pixels < min_pixels).
    [OUTPUT]: Returns n_pixels = 0 and NaN for error measures.
    """
    # --- [1. INPUT] ---
    img_a = np.ones((30, 30, 3), dtype=np.float32)
    m_a = np.zeros((30, 30), dtype=bool); m_a[:, :10] = True
    img_b = np.ones((30, 30, 3), dtype=np.float32)
    m_b = np.zeros((30, 30), dtype=bool); m_b[:, 20:] = True  # Disjoint

    # --- [2. ACT] ---
    metrics = overlap_photometric_error(img_a, m_a, img_b, m_b, erode=1, min_pixels=10)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert metrics["n_pixels"] == 0
    assert np.isnan(metrics["mae"])


def test_homography_rms_error():
    """
    [INPUT]:
        - H_true: Identity homography.
        - H_est: Homography with a pure translation of dx = 3.0, dy = 4.0.
        - shape: (100, 100).
    [OUTPUT]: Corner RMS error must equal sqrt(3^2 + 4^2) = 5.0 pixels.
    """
    # --- [1. INPUT] ---
    H_true = np.eye(3)
    H_est = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    shape = (100, 100)

    # --- [2. ACT] ---
    rms_err = homography_rms_error(H_est, H_true, shape)
    rms_bounded = homography_rms_error(H_est, H_true, shape, dst_shape=(80, 80))

    # --- [3. OUTPUT ASSERTIONS] ---
    assert np.isclose(rms_err, 5.0, atol=1e-5), f"Expected corner RMSE 5.0, got {rms_err}"
    assert np.isclose(rms_bounded, 5.0, atol=1e-5)


def test_overlap_ncc_bands():
    """
    [INPUT]: Two 60x60 images with vertical overlap and textured pattern.
    [OUTPUT]: overlap_ncc_bands returns list of n_bands floats close to 1.0.
    """
    # --- [1. INPUT] ---
    np.random.seed(42)
    img = np.random.uniform(0.1, 0.9, size=(60, 60, 3)).astype(np.float32)
    mask = np.ones((60, 60), dtype=bool)

    # --- [2. ACT] ---
    bands = overlap_ncc_bands(img, mask, img, mask, n_bands=4, erode=0, min_pixels=50)
    empty_bands = overlap_ncc_bands(img, np.zeros_like(mask), img, mask, n_bands=4)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(bands) == 4
    for val in bands:
        assert np.isclose(val, 1.0, atol=1e-3) or np.isnan(val)
    assert len(empty_bands) == 4
    assert all(np.isnan(b) for b in empty_bands)


def test_homography_rms_error_empty_destination_overlap():
    """
    [INPUT]: Ground-truth homography mapping points far away outside dst_shape bounds.
    [OUTPUT]: homography_rms_error returns NaN when zero evaluation points fall inside dst_shape.
    """
    # --- [1. INPUT] ---
    H_far = np.array([[1.0, 0.0, 800.0], [0.0, 1.0, 800.0], [0.0, 0.0, 1.0]])
    H_est = np.eye(3)

    # --- [2. ACT] ---
    err = homography_rms_error(H_est, H_far, image_shape=(50, 50), dst_shape=(50, 50))

    # --- [3. OUTPUT ASSERTIONS] ---
    assert np.isnan(err)


def test_overlap_photometric_error_no_erosion():
    """
    [INPUT]: Two images evaluated with erode = 0 (skipping erosion branch).
    [OUTPUT]: Metrics computed directly on un-eroded intersection mask.
    """
    # --- [1. INPUT] ---
    img = np.ones((20, 20, 3), dtype=np.float32) * 0.5
    mask = np.ones((20, 20), dtype=bool)

    # --- [2. ACT] ---
    metrics = overlap_photometric_error(img, mask, img, mask, erode=0, min_pixels=50)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert metrics["n_pixels"] == 400
    assert np.isclose(metrics["mae"], 0.0)
