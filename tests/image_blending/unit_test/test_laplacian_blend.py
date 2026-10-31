"""
Unit tests for image_blending.laplacian_blend module following the Input-Output testing model.
"""

import cv2
import numpy as np
import pytest
from image_blending.laplacian_blend import (
    _conv,
    _GAUSS5,
    pyr_down,
    pyr_up,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    laplacian_blend,
)


def test_gauss5_kernel_properties():
    """
    [INPUT]: Standard 5-tap Gaussian binomial kernel _GAUSS5: [1, 4, 6, 4, 1] / 16.
    [OUTPUT]: Kernel elements must sum to 1.0 (energy conserving) and be symmetric.
    """
    # --- [1. INPUT] ---
    kernel = _GAUSS5

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    assert np.isclose(np.sum(kernel), 1.0), "Gaussian filter must be normalized to sum to 1.0"
    np.testing.assert_array_equal(kernel, kernel[::-1], err_msg="Gaussian filter must be symmetric.")


def test_conv_constant_signal_preservation():
    """
    [INPUT]: Constant 16x16x3 image with pixel value 0.75.
    [OUTPUT]: Convolving with normalized Gaussian filter must leave pixel values unchanged at 0.75.
    """
    # --- [1. INPUT] ---
    input_image = np.full((16, 16, 3), 0.75, dtype=np.float64)

    # --- [2. ACT] ---
    actual_output = _conv(input_image, _GAUSS5)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(actual_output, input_image, rtol=1e-4, err_msg="DC signal was not preserved by convolution.")


def test_pyr_down_and_pyr_up_shapes():
    """
    [INPUT]: 16x16x3 float32 image.
    [OUTPUT]:
        - pyr_down must halve dimensions to (8, 8, 3).
        - pyr_up must double dimensions back to (16, 16, 3).
    """
    # --- [1. INPUT] ---
    input_image = np.ones((16, 16, 3), dtype=np.float32)

    # --- [2. ACT] ---
    downsampled_output = pyr_down(input_image)
    upsampled_output = pyr_up(downsampled_output)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert downsampled_output.shape == (8, 8, 3), f"Expected shape (8, 8, 3), got {downsampled_output.shape}"
    assert upsampled_output.shape == (16, 16, 3), f"Expected shape (16, 16, 3), got {upsampled_output.shape}"


def test_pyr_up_grayscale():
    """
    [INPUT]: 2D single-channel (grayscale) image of shape (12, 12).
    [OUTPUT]: pyr_up must correctly handle 2D inputs and produce shape (24, 24).
    """
    # --- [1. INPUT] ---
    input_gray = np.ones((12, 12), dtype=np.float32)

    # --- [2. ACT] ---
    actual_output = pyr_up(input_gray)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_output.shape == (24, 24), f"Expected shape (24, 24), got {actual_output.shape}"


def test_pyramid_hierarchy_lengths():
    """
    [INPUT]: 32x32x3 base image, levels = 3.
    [OUTPUT]:
        - Gaussian pyramid must contain 4 levels with shapes (32, 32), (16, 16), (8, 8), (4, 4).
        - Laplacian pyramid must contain 4 levels from coarsest (4, 4) to finest (32, 32).
    """
    # --- [1. INPUT] ---
    input_image = np.ones((32, 32, 3), dtype=np.float32)
    levels = 3

    # --- [2. ACT] ---
    gp_output = build_gaussian_pyramid(input_image, levels=levels)
    lp_output = build_laplacian_pyramid(gp_output, levels=levels)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(gp_output) == levels + 1, f"Expected {levels + 1} GP levels, got {len(gp_output)}"
    assert gp_output[0].shape == (32, 32, 3), "GP level 0 must match original size"
    assert gp_output[3].shape == (4, 4, 3), "GP coarsest level must be 4x4x3"

    assert len(lp_output) == levels + 1, f"Expected {levels + 1} LP levels, got {len(lp_output)}"
    assert lp_output[0].shape == (4, 4, 3), "LP level 0 must be coarsest representation"
    assert lp_output[3].shape == (32, 32, 3), "LP finest level must match original size"


def test_laplacian_pyramid_reconstruction_fidelity():
    """
    [INPUT]: Random continuous test image in range [0.1, 0.9], levels = 2.
    [OUTPUT]: Collapsing the Laplacian pyramid must reconstruct the original image within numerical tolerance (atol < 0.08).
    """
    # --- [1. INPUT] ---
    np.random.seed(123)
    input_image = np.random.uniform(0.1, 0.9, size=(32, 32, 3)).astype(np.float32)
    levels = 2

    # --- [2. ACT] ---
    gp = build_gaussian_pyramid(input_image, levels=levels)
    lp = build_laplacian_pyramid(gp, levels=levels)

    reconstructed_output = lp[0].copy()
    for i in range(1, levels + 1):
        reconstructed_output = cv2.pyrUp(reconstructed_output)
        reconstructed_output = cv2.resize(reconstructed_output, (lp[i].shape[1], lp[i].shape[0]))
        reconstructed_output += lp[i]

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(
        reconstructed_output, input_image, atol=0.08,
        err_msg="Laplacian pyramid reconstruction failed mathematical fidelity check."
    )


def test_laplacian_blend_execution():
    """
    [INPUT]:
        - Target: 32x32x3 constant 0.2 array.
        - Source: 16x16x3 constant 0.8 array.
        - Mask: 16x16x3 with central 8x8 active (1.0).
        - Dims: Placement ROI (8, 24, 8, 24), num_levels = 2.
    [OUTPUT]:
        - Output shape matches target shape (32, 32, 3).
        - Values strictly bounded in [0.0, 1.0] without NaNs.
        - Outside boundary regions remain identical to target.
    """
    # --- [1. INPUT] ---
    input_target = np.full((32, 32, 3), 0.2, dtype=np.float32)
    input_source = np.full((16, 16, 3), 0.8, dtype=np.float32)
    input_mask = np.zeros((16, 16, 3), dtype=np.float32)
    input_mask[4:12, 4:12, :] = 1.0
    input_dims = (8, 24, 8, 24)

    input_data = {
        "source": input_source,
        "mask": input_mask,
        "target": input_target,
        "dims": input_dims,
    }

    # --- [2. ACT] ---
    actual_output = laplacian_blend(input_data, num_levels=2)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_output.shape == input_target.shape, "Output shape must equal target shape"
    assert actual_output.min() >= 0.0 and actual_output.max() <= 1.0, "Output pixels out of range [0, 1]"
    assert not np.isnan(actual_output).any(), "Output contains NaN values"
    np.testing.assert_allclose(
        actual_output[:8, :, :], input_target[:8, :, :], atol=1e-5,
        err_msg="Target area outside ROI was modified."
    )


def test_laplacian_blend_2d_mask():
    """
    [INPUT]: 2D single-channel mask of shape (16, 16).
    [OUTPUT]: laplacian_blend must expand the 2D mask to 3 channels and return valid blended image.
    """
    # --- [1. INPUT] ---
    input_target = np.full((32, 32, 3), 0.3, dtype=np.float32)
    input_source = np.full((16, 16, 3), 0.9, dtype=np.float32)
    input_mask_2d = np.zeros((16, 16), dtype=np.float32)
    input_mask_2d[4:12, 4:12] = 1.0
    input_dims = (8, 24, 8, 24)

    input_data = {
        "source": input_source,
        "mask": input_mask_2d,
        "target": input_target,
        "dims": input_dims,
    }

    # --- [2. ACT] ---
    actual_output = laplacian_blend(input_data, num_levels=2)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_output.shape == (32, 32, 3), "Failed to blend using 2D mask"
    assert actual_output.min() >= 0.0 and actual_output.max() <= 1.0
