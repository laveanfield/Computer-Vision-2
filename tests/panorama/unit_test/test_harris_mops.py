"""
Unit tests for panorama.harris_mops module following the Input-Output testing model.
"""

import numpy as np
import pytest
from panorama.harris_mops import (
    harris_response,
    _peaks,
    anms,
    detect_harris_mops,
)


def test_harris_response_flat_image():
    """
    [INPUT]: Uniform flat image (pixel values constant at 128.0).
    [OUTPUT]: Response map must be all zeros (no gradient energy, det=0, tr=0).
    """
    # --- [1. INPUT] ---
    img_flat = np.full((50, 50), 128.0, dtype=np.float32)

    # --- [2. ACT] ---
    resp_harmonic = harris_response(img_flat, mode="harmonic")
    resp_harris = harris_response(img_flat, mode="harris")

    # --- [3. OUTPUT ASSERTIONS] ---
    assert np.allclose(resp_harmonic, 0.0), "Flat image must produce zero harmonic corner response"
    assert np.allclose(resp_harris, 0.0), "Flat image must produce zero standard Harris response"


def test_harris_response_corner_detection(synthetic_corner_image):
    """
    [INPUT]: synthetic_corner_image containing rectangular corners.
    [OUTPUT]: Corner strength map f has distinct high response peaks at corner locations.
    """
    # --- [1. INPUT] ---
    img = synthetic_corner_image.astype(np.float32) / 255.0

    # --- [2. ACT] ---
    resp = harris_response(img, mode="harmonic")

    # --- [3. OUTPUT ASSERTIONS] ---
    assert resp.shape == img.shape
    assert resp.max() > 0.001, "Expected strong response at geometric corners"


def test_peaks_extraction():
    """
    [INPUT]: Synthetic 80x80 response map with one known isolated maximum at (40, 40) (beyond MARGIN=27).
    [OUTPUT]: _peaks identifies (x=40, y=40) with sub-pixel refinement near 40.0.
    """
    # --- [1. INPUT] ---
    resp = np.zeros((80, 80), dtype=np.float32)
    resp[40, 40] = 5.0
    resp[39, 40] = 2.0; resp[41, 40] = 2.0
    resp[40, 39] = 2.0; resp[40, 41] = 2.0

    # --- [2. ACT] ---
    xs, ys, scores = _peaks(resp, thresh_rel=0.1, max_candidates=10)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(xs) == 1
    assert np.isclose(xs[0], 40.0, atol=0.2)
    assert np.isclose(ys[0], 40.0, atol=0.2)
    assert scores[0] > 0.0


def test_anms_dispersion():
    """
    [INPUT]:
        - 10 clustered points around (5, 5).
        - 2 distant points at (50, 50) and (80, 80).
        - Requesting max_keep = 3.
    [OUTPUT]: ANMS retains points that maximize spatial separation instead of clustering at (5, 5).
    """
    # --- [1. INPUT] ---
    cluster_x = [5.0 + 0.1 * i for i in range(10)]
    cluster_y = [5.0 + 0.1 * i for i in range(10)]
    cluster_s = [10.0 + i for i in range(10)]

    xs = np.array(cluster_x + [50.0, 80.0], dtype=np.float32)
    ys = np.array(cluster_y + [50.0, 80.0], dtype=np.float32)
    scores = np.array(cluster_s + [8.0, 9.0], dtype=np.float32)

    # --- [2. ACT] ---
    xy = np.stack([xs, ys], axis=1)
    keep_indices = anms(xy, scores, n_keep=3)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(keep_indices) == 3
    # Distant points (indices 10 or 11) must be preserved due to high suppression radii
    assert (10 in keep_indices) or (11 in keep_indices), "ANMS failed to keep spatially distant keypoints."


def test_detect_harris_mops_pipeline(synthetic_corner_image):
    """
    [INPUT]: synthetic_corner_image, max_keypoints = 50.
    [OUTPUT]:
        - Keypoints array with <= 50 keypoints, shape (N, 2).
        - Descriptors shape is (N, 64) with float32 values.
        - Descriptors have approximately zero mean and unit variance per patch.
    """
    # --- [1. INPUT] ---
    img_gray = synthetic_corner_image
    max_kpts = 50

    # --- [2. ACT] ---
    pts, desc, scale, resp = detect_harris_mops(img_gray, max_keypoints=max_kpts, levels=3)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(pts) > 0 and len(pts) <= max_kpts
    assert pts.shape[1] == 2
    assert desc.shape == (len(pts), 64)
    assert desc.dtype == np.float32

    # Check zero-mean unit variance of normalized MOPS patches
    means = desc.mean(axis=1)
    stds = desc.std(axis=1)
    np.testing.assert_allclose(means, 0.0, atol=1e-2, err_msg="MOPS patches not zero-mean normalized.")
    np.testing.assert_allclose(stds, 1.0, atol=1e-2, err_msg="MOPS patches not unit-variance normalized.")


def test_peaks_max_candidates_truncation():
    """
    [INPUT]: 80x80 response map with 5 separate peaks, but max_candidates = 2.
    [OUTPUT]: _peaks returns exactly 2 peaks (exercising argpartition truncation).
    """
    # --- [1. INPUT] ---
    resp = np.zeros((80, 80), dtype=np.float32)
    peak_coords = [(35, 35), (40, 40), (45, 45), (50, 50), (40, 50)]
    for y, x in peak_coords:
        resp[y, x] = 5.0
        resp[y - 1, x] = 2.0; resp[y + 1, x] = 2.0
        resp[y, x - 1] = 2.0; resp[y, x + 1] = 2.0

    # --- [2. ACT] ---
    xs, ys, scores = _peaks(resp, thresh_rel=0.1, max_candidates=2)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(xs) == 2
    assert len(ys) == 2


def test_detect_harris_mops_empty_peaks():
    """
    [INPUT]: Small float32 image (40x40, min dimension < 2*MARGIN+3=57) with dtype=np.float32.
    [OUTPUT]: detect_harris_mops breaks immediately (xs is empty) and returns empty arrays (shape (0, 2) and (0, 64)).
    """
    # --- [1. INPUT] ---
    img_small_float = np.zeros((40, 40), dtype=np.float32)

    # --- [2. ACT] ---
    pts, desc, w_size, strength = detect_harris_mops(img_small_float)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(pts) == 0
    assert pts.shape == (0, 2)
    assert desc.shape == (0, 64)


def test_detect_harris_mops_full_loop_natural_exit():
    """
    [INPUT]: 200x200 image with levels=1 (dimension >= 2*MARGIN+3=57, avoiding break).
    [OUTPUT]: Loop over pyramid levels terminates naturally (exercising branch 199->204).
    """
    import cv2
    # --- [1. INPUT] ---
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (70, 70), (130, 130), 255, -1)

    # --- [2. ACT] ---
    pts, desc, w_size, s = detect_harris_mops(img, levels=1)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(pts) > 0
    assert pts.shape[1] == 2
