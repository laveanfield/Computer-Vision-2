"""
Unit tests for panorama.synthetic module following the Input-Output testing model.
"""

import cv2
from unittest.mock import patch
import numpy as np
import pytest
from panorama.synthetic import (
    intrinsics,
    rotation_matrix,
    ViewSpec,
    render_views,
    overlap_fraction,
    nominal_overlap,
    sweep_pair,
    _NoJitter,
)


def test_intrinsics_matrix():
    """
    [INPUT]: Focal length f = 800.0, width = 640, height = 480.
    [OUTPUT]: 3x3 camera matrix K with optical center cx = 319.5, cy = 239.5.
    """
    # --- [1. INPUT] ---
    f, w, h = 800.0, 640, 480

    # --- [2. ACT] ---
    K = intrinsics(f, w, h)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert K.shape == (3, 3)
    assert K[0, 0] == f and K[1, 1] == f
    assert K[0, 2] == (w - 1) / 2.0
    assert K[1, 2] == (h - 1) / 2.0
    assert K[2, 2] == 1.0


def test_rotation_matrix_properties():
    """
    [INPUT]: Euler angles yaw=10, pitch=5, roll=-15.
    [OUTPUT]:
        - R must belong to SO(3): det(R) == 1.0 and R @ R.T == Identity.
        - Zero angles must yield Identity matrix.
    """
    # --- [1. INPUT] ---
    R_zero = rotation_matrix(yaw=0.0, pitch=0.0, roll=0.0)
    R_rot = rotation_matrix(yaw=10.0, pitch=5.0, roll=-15.0)

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(R_zero, np.eye(3), atol=1e-7, err_msg="Zero rotation must equal Identity")
    np.testing.assert_allclose(R_rot @ R_rot.T, np.eye(3), atol=1e-7, err_msg="Rotation matrix must be orthogonal")
    assert np.isclose(np.linalg.det(R_rot), 1.0), "Determinant of rotation matrix must be 1.0"


def test_render_views_and_ground_truth(synthetic_corner_image):
    """
    [INPUT]: Base image (synthetic_corner_image, converted to BGR), 2 ViewSpecs with yaw=0 and yaw=4.
    [OUTPUT]:
        - 2 rendered views of size (100, 100, 3) uint8.
        - Ground truth homography H_gt[(0, 1)] and H_gt[(1, 0)] normalized to H[2, 2] == 1.0.
        - Coverage values in [0, 1].
    """
    # --- [1. INPUT] ---
    world_bgr = cv2.cvtColor(synthetic_corner_image, cv2.COLOR_GRAY2BGR)
    specs = [
        ViewSpec(yaw=0.0, pitch=0.0),
        ViewSpec(yaw=4.0, pitch=0.0),
    ]

    # --- [2. ACT] ---
    views, H_gt, cov = render_views(world_bgr, specs, size=(100, 100), f_world=300.0, seed=42)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(views) == 2
    assert views[0].shape == (100, 100, 3)
    assert views[1].shape == (100, 100, 3)
    assert (0, 1) in H_gt and (1, 0) in H_gt
    assert np.isclose(H_gt[(0, 1)][2, 2], 1.0)
    assert len(cov) == 2
    assert all(0.0 <= c <= 1.0 for c in cov)


def test_overlap_fraction_and_nominal():
    """
    [INPUT]: Identity homography matrix and canvas shape (100, 100).
    [OUTPUT]:
        - overlap_fraction is 1.0.
        - nominal_overlap for 6 degrees overlap returns a positive fraction in (0.0, 1.0].
    """
    # --- [1. INPUT] ---
    H_ident = np.eye(3, dtype=np.float64)
    shape = (100, 100)

    # --- [2. ACT] ---
    frac_ident = overlap_fraction(H_ident, shape, shape)
    nom_ov = nominal_overlap("overlap", 6.0)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert np.isclose(frac_ident, 1.0)
    assert 0.0 < nom_ov <= 1.0


def test_sweep_pair_generation():
    """
    [INPUT]: sweeps ('overlap', 'roll', 'scale', 'noise', 'exposure', 'blur') with _NoJitter().
    [OUTPUT]: Returns tuple of two ViewSpec instances with correctly populated fields.
    """
    # --- [1. INPUT & 2. ACT] ---
    v_a, v_b = sweep_pair("overlap", 12.0, _NoJitter())
    r_a, r_b = sweep_pair("roll", 45.0, _NoJitter())
    s_a, s_b = sweep_pair("scale", 2.0, _NoJitter())
    n_a, n_b = sweep_pair("noise", 20.0, _NoJitter())
    e_a, e_b = sweep_pair("exposure", 0.5, _NoJitter())
    b_a, b_b = sweep_pair("blur", 2.0, _NoJitter())

    # --- [3. OUTPUT ASSERTIONS] ---
    assert v_a.yaw == -6.0 and v_b.yaw == 6.0
    assert r_b.roll == 45.0
    assert s_b.f is not None
    assert n_a.noise == 20.0 and n_b.noise == 20.0
    assert e_b.gain == 0.5
    assert b_b.blur == 2.0

    with pytest.raises(ValueError, match="unknown sweep"):
        sweep_pair("unknown_sweep_key", 1.0, _NoJitter())


def test_render_views_with_blur_and_noise(synthetic_corner_image):
    """
    [INPUT]: ViewSpecs with blur > 0 and noise > 0.
    [OUTPUT]: Returns blurred and noisy uint8 images.
    """
    # --- [1. INPUT] ---
    world_bgr = cv2.cvtColor(synthetic_corner_image, cv2.COLOR_GRAY2BGR)
    specs = [
        ViewSpec(yaw=0.0, blur=1.5, noise=5.0),
    ]

    # --- [2. ACT] ---
    views, _, _ = render_views(world_bgr, specs, size=(80, 80), seed=1)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(views) == 1
    assert views[0].shape == (80, 80, 3)
    assert views[0].dtype == np.uint8


def test_sweep_pair_custom_extension():
    """
    [INPUT]: SWEEPS dictionary augmented with a custom sweep key.
    [OUTPUT]: sweep_pair falls through to default return branch (exercising line 261->263).
    """
    with patch.dict("panorama.synthetic.SWEEPS", {"custom_sweep": ("Custom", [1.0])}):
        va, vb = sweep_pair("custom_sweep", 1.0, _NoJitter())
        assert isinstance(va, ViewSpec) and isinstance(vb, ViewSpec)
