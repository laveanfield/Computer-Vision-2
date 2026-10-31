"""
Unit tests for panorama.matching module following the Input-Output testing model.
"""

import numpy as np
import pytest
from panorama.features import Features
from panorama.matching import _prepare, knn2, match_features, Matches


def test_prepare_float_and_binary():
    """
    [INPUT]:
        - Float descriptors: 2x4 array.
        - Binary descriptors: 2x1 uint8 array (e.g. byte 0b10100000 = 160).
    [OUTPUT]:
        - Float descriptors remain float32 array of shape (2, 4).
        - Binary descriptors unpacked into 8 float32 bits per byte -> shape (2, 8).
    """
    # --- [1. INPUT] ---
    float_desc = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    binary_desc = np.array([[160], [255]], dtype=np.uint8)

    # --- [2. ACT] ---
    p_float = _prepare(float_desc, binary=False)
    p_binary = _prepare(binary_desc, binary=True)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert p_float.shape == (2, 2) and p_float.dtype == np.float32
    assert p_binary.shape == (2, 8) and p_binary.dtype == np.float32
    np.testing.assert_array_equal(p_binary[0], [1, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(p_binary[1], [1, 1, 1, 1, 1, 1, 1, 1])


def test_knn2_exact_nearest_neighbors():
    """
    [INPUT]:
        - Query desc1: [[1.0, 0.0], [0.0, 1.0]].
        - Train desc2: [[1.0, 0.1], [1.0, 0.9], [0.0, 0.8]].
    [OUTPUT]:
        - Query 0's 1st neighbor is index 0 (dist 0.1), 2nd neighbor is index 1.
        - Query 1's 1st neighbor is index 2 (dist 0.2), 2nd neighbor is index 1.
    """
    # --- [1. INPUT] ---
    desc1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    desc2 = np.array([[1.0, 0.1], [1.0, 0.9], [0.0, 0.8]], dtype=np.float32)

    # --- [2. ACT] ---
    indices, distances = knn2(desc1, desc2, binary=False)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert indices.shape == (2, 2)
    assert distances.shape == (2, 2)
    assert indices[0, 0] == 0 and indices[1, 0] == 2
    assert distances[0, 0] < distances[0, 1]


def test_match_features_lowe_ratio_filtering():
    """
    [INPUT]:
        - Feature 1 has a distinct nearest neighbor (d1 / d2 = 0.2 < 0.75).
        - Feature 2 has an ambiguous nearest neighbor (d1 / d2 = 0.9 > 0.75).
    [OUTPUT]:
        - Only Feature 1 survives Lowe's ratio test; total matches = 1.
    """
    # --- [1. INPUT] ---
    f1 = Features(
        method="sift",
        kpts=np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
        desc=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        response=np.ones(2, dtype=np.float32),
        binary=False,
        seconds=0.01,
    )
    f2 = Features(
        method="sift",
        kpts=np.array([[10.0, 10.0], [50.0, 50.0], [20.0, 20.0]], dtype=np.float32),
        desc=np.array([
            [0.9, 0.0, 0.0],   # Close match for f1[0] (dist 0.1, 2nd dist > 1.0 -> distinct!)
            [0.0, 0.96, 0.0],  # 1st candidate for f1[1] (dist 0.04)
            [0.0, 0.95, 0.0],  # 2nd candidate for f1[1] (dist 0.05 -> ratio 0.04/0.05 = 0.80 > 0.75 -> ambiguous!)
        ], dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
        response=np.ones(3, dtype=np.float32),
        binary=False,
        seconds=0.01,
    )

    # --- [2. ACT] ---
    matches = match_features(f1, f2, ratio=0.75, mutual=False, unique=False)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(matches) == 1
    assert matches.idx1[0] == 0
    assert matches.idx2[0] == 0


def test_match_features_empty_features():
    """
    [INPUT]: One or both Features objects having 0 keypoints.
    [OUTPUT]: Returns empty Matches object (len == 0) without exceptions.
    """
    # --- [1. INPUT] ---
    f_empty = Features("sift", np.empty((0, 2)), np.empty((0, 128)), np.empty(0), np.empty(0), False, 0.0)
    f_valid = Features("sift", np.ones((5, 2)), np.ones((5, 128)), np.ones(5), np.ones(5), False, 0.0)

    # --- [2. ACT] ---
    m1 = match_features(f_empty, f_valid)
    m2 = match_features(f_valid, f_empty)
    m3 = match_features(f_empty, f_empty)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(m1) == 0 and len(m2) == 0 and len(m3) == 0


def test_match_features_unique_enforcement():
    """
    [INPUT]:
        - Two query features both match the same train feature with ratio < 0.75.
        - unique=True.
    [OUTPUT]: Enforces 1-to-1 matching, keeping only the query feature with smaller distance.
    """
    # --- [1. INPUT] ---
    f1 = Features(
        "sift",
        kpts=np.array([[5.0, 5.0], [6.0, 6.0]], dtype=np.float32),
        desc=np.array([[1.0, 0.0], [0.9, 0.0]], dtype=np.float32),
        scale=np.ones(2), response=np.ones(2), binary=False, seconds=0.01
    )
    f2 = Features(
        "sift",
        kpts=np.array([[10.0, 10.0], [90.0, 90.0]], dtype=np.float32),
        desc=np.array([[1.0, 0.0], [0.0, 10.0]], dtype=np.float32),
        scale=np.ones(2), response=np.ones(2), binary=False, seconds=0.01
    )

    # --- [2. ACT] ---
    matches = match_features(f1, f2, ratio=0.8, unique=True)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(matches) == 1
    assert matches.idx1[0] == 0, "Query 0 (distance 0.0) should beat Query 1 (distance 0.1)"


def test_knn2_single_train_descriptor():
    """
    [INPUT]: Query descriptors with 2 rows, but train descriptors with only 1 row (< 2).
    [OUTPUT]: knn2 returns inf distances and zeros indices.
    """
    # --- [1. INPUT] ---
    desc1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    desc2 = np.array([[1.0, 2.0]], dtype=np.float32)  # only 1 row

    # --- [2. ACT] ---
    idx, dist = knn2(desc1, desc2, binary=False)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert idx.shape == (2, 2)
    assert np.all(np.isneginf(dist) == False)
    assert np.all(np.isposinf(dist))


def test_match_features_type_mismatch():
    """
    [INPUT]: f1 with binary=True (ORB) and f2 with binary=False (SIFT).
    [OUTPUT]: Raises ValueError detailing binary and float descriptor mismatch.
    """
    # --- [1. INPUT] ---
    f_orb = Features("orb", np.zeros((2, 2)), np.zeros((2, 32), dtype=np.uint8), np.ones(2), np.ones(2), True, 0.01)
    f_sift = Features("sift", np.zeros((2, 2)), np.zeros((2, 128), dtype=np.float32), np.ones(2), np.ones(2), False, 0.01)

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    with pytest.raises(ValueError, match="Cannot match binary and float"):
        match_features(f_orb, f_sift)


def test_match_features_mutual_consistency():
    """
    [INPUT]: Two feature sets matched with mutual=True.
    [OUTPUT]: Returns consistent cross-checked matches.
    """
    # --- [1. INPUT] ---
    f1 = Features(
        "sift",
        kpts=np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
        desc=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        scale=np.ones(2), response=np.ones(2), binary=False, seconds=0.01
    )
    f2 = Features(
        "sift",
        kpts=np.array([[10.0, 10.0], [20.0, 20.0], [90.0, 90.0]], dtype=np.float32),
        desc=np.array([[0.95, 0.0], [0.0, 0.95], [5.0, 5.0]], dtype=np.float32),
        scale=np.ones(3), response=np.ones(3), binary=False, seconds=0.01
    )

    # --- [2. ACT] ---
    matches = match_features(f1, f2, ratio=0.8, mutual=True)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(matches) == 2
