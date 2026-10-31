"""
Unit tests for panorama.warp module following the Input-Output testing model.
"""

import numpy as np
import pytest
from panorama.warp import bilinear_sample, compute_canvas, warp_perspective


def test_bilinear_sample_integer_coordinates():
    """
    [INPUT]: Constant gradient image 4x4, sampling at integer grid positions (x=1, y=2).
    [OUTPUT]: Sampled value must strictly match array element img[2, 1] with valid=True.
    """
    # --- [1. INPUT] ---
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    x = np.array([1.0])
    y = np.array([2.0])
    expected_val = img[2, 1]  # row 2, col 1 = 9.0

    # --- [2. ACT] ---
    sampled_vals, valid_mask = bilinear_sample(img, x, y)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert valid_mask[0] is True or valid_mask[0] == 1
    assert np.isclose(sampled_vals[0], expected_val)


def test_bilinear_sample_half_pixel_interpolation():
    """
    [INPUT]: 2x2 image [[0, 10], [20, 30]], sampling at center (x=0.5, y=0.5).
    [OUTPUT]: Returns average of 4 pixels = (0 + 10 + 20 + 30) / 4 = 15.0.
    """
    # --- [1. INPUT] ---
    img = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    x = np.array([0.5])
    y = np.array([0.5])
    expected_val = 15.0

    # --- [2. ACT] ---
    sampled_vals, valid_mask = bilinear_sample(img, x, y)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert valid_mask[0] is True or valid_mask[0] == 1
    assert np.isclose(sampled_vals[0], expected_val)


def test_bilinear_sample_out_of_bounds():
    """
    [INPUT]: 10x10 image with out-of-bounds queries: x = -1.0, y = 15.0.
    [OUTPUT]: valid_mask must be False and returned pixel value must be 0.0.
    """
    # --- [1. INPUT] ---
    img = np.ones((10, 10), dtype=np.float32)
    x = np.array([-1.0, 5.0])
    y = np.array([5.0, 15.0])

    # --- [2. ACT] ---
    sampled_vals, valid_mask = bilinear_sample(img, x, y)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert not valid_mask.any(), "Points outside [0, W-1] x [0, H-1] must have valid=False"
    np.testing.assert_allclose(sampled_vals, [0.0, 0.0])


def test_bilinear_sample_color_channels():
    """
    [INPUT]: 3-channel color image of shape (8, 8, 3).
    [OUTPUT]: Returns 3-channel interpolated samples with shape matching (N, 3).
    """
    # --- [1. INPUT] ---
    img_rgb = np.full((8, 8, 3), [1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([2.5, 3.0])
    y = np.array([2.5, 4.0])

    # --- [2. ACT] ---
    sampled_rgb, valid_mask = bilinear_sample(img_rgb, x, y)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert sampled_rgb.shape == (2, 3), f"Expected shape (2, 3), got {sampled_rgb.shape}"
    assert valid_mask.all()
    np.testing.assert_allclose(sampled_rgb[0], [1.0, 2.0, 3.0])


def test_compute_canvas_bounds():
    """
    [INPUT]:
        - Two 50x50 image shapes.
        - Homographies: Identity (image 0) and Translation [40, 0] (image 1).
    [OUTPUT]:
        - Global canvas bounding box covering width 50 + 40 = 90, height 50.
        - Translation matrix T correctly positioning the origin.
    """
    # --- [1. INPUT] ---
    shapes = [(50, 50), (50, 50)]
    H0 = np.eye(3)
    H1 = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # --- [2. ACT] ---
    (canvas_h, canvas_w), T = compute_canvas(shapes, [H0, H1], max_side=None)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert canvas_w == 90, f"Expected canvas width 90, got {canvas_w}"
    assert canvas_h == 50, f"Expected canvas height 50, got {canvas_h}"
    assert isinstance(T, np.ndarray) and T.shape == (3, 3)


def test_warp_perspective_identity_reconstruction():
    """
    [INPUT]: 30x30 test pattern warped using Identity homography H = np.eye(3).
    [OUTPUT]: Output warped image matches original exactly inside the valid mask.
    """
    # --- [1. INPUT] ---
    img = np.random.uniform(0.1, 0.9, size=(30, 30, 3)).astype(np.float32)
    H_id = np.eye(3)

    # --- [2. ACT] ---
    warped, mask = warp_perspective(img, H_id, (30, 30))

    # --- [3. OUTPUT ASSERTIONS] ---
    assert warped.shape == (30, 30, 3)
    assert mask.shape == (30, 30)
    assert mask.all(), "Identity warp should cover entire canvas"
    np.testing.assert_allclose(warped, img, atol=1e-5)


def test_warp_perspective_disjoint_footprint():
    """
    [INPUT]: 20x20 image warped with a large translation (800, 800) into a small 50x50 canvas.
    [OUTPUT]: Returns empty all-zeros array and all-False mask (exercising x1 <= x0 or y1 <= y0).
    """
    # --- [1. INPUT] ---
    img = np.ones((20, 20, 3), dtype=np.float32)
    H_far = np.array([[1.0, 0.0, 800.0], [0.0, 1.0, 800.0], [0.0, 0.0, 1.0]])

    # --- [2. ACT] ---
    warped, mask = warp_perspective(img, H_far, (50, 50))

    # --- [3. OUTPUT ASSERTIONS] ---
    assert warped.shape == (50, 50, 3)
    assert not mask.any()
    assert np.allclose(warped, 0.0)


def test_warp_perspective_nan_corners():
    """
    [INPUT]: Degenerate homography mapping corner coordinates to infinity/NaN.
    [OUTPUT]: Returns zeros array and empty mask without throwing exceptions.
    """
    # --- [1. INPUT] ---
    img = np.ones((20, 20, 3), dtype=np.float32)
    H_nan = np.zeros((3, 3))

    # --- [2. ACT] ---
    warped, mask = warp_perspective(img, H_nan, (50, 50))

    # --- [3. OUTPUT ASSERTIONS] ---
    assert not mask.any()
    assert np.allclose(warped, 0.0)


def test_compute_canvas_degenerate_infinity():
    """
    [INPUT]: Homography mapping an image corner to infinity (w = 0).
    [OUTPUT]: Raises ValueError detailing degenerate projection.
    """
    # --- [1. INPUT] ---
    shapes = [(50, 50)]
    H_deg = np.zeros((3, 3))

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    with pytest.raises(ValueError, match="sends an image corner to infinity"):
        compute_canvas(shapes, [H_deg])


def test_compute_canvas_max_side_scaling():
    """
    [INPUT]: Two shapes requiring a 500px wide canvas, but max_side constraint is 100.
    [OUTPUT]: Canvas width is scaled down to <= 100 with matching scale matrix T.
    """
    # --- [1. INPUT] ---
    shapes = [(100, 100), (100, 100)]
    H0 = np.eye(3)
    H1 = np.array([[1.0, 0.0, 400.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # --- [2. ACT] ---
    (canvas_h, canvas_w), T = compute_canvas(shapes, [H0, H1], max_side=100)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert max(canvas_h, canvas_w) <= 100
    assert T[0, 0] < 1.0  # Scale factor applied
