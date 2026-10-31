"""
Unit tests for image_blending.naive_blend module following the Input-Output testing model.
"""

import numpy as np
import pytest
from image_blending.naive_blend import naive_blend


def test_naive_blend_full_mask():
    """
    [INPUT]:
        - Target: 10x10x3 zeros (black background).
        - Source: 4x4x3 constant array of 0.8.
        - Mask: 4x4x3 all 1.0 (100% alpha source opacity).
        - Dims: Bounding box ROI (2, 6, 2, 6).
    [OUTPUT]:
        - The target region [2:6, 2:6] must equal the source image exactly (value 0.8).
    """
    # --- [1. INPUT] ---
    input_target = np.zeros((10, 10, 3), dtype=np.float32)
    input_source = np.full((4, 4, 3), 0.8, dtype=np.float32)
    input_mask = np.ones((4, 4, 3), dtype=np.float32)
    input_dims = (2, 6, 2, 6)

    input_data = {
        "target": input_target,
        "source": input_source,
        "mask": input_mask,
        "dims": input_dims,
    }

    # --- [2. ACT] ---
    actual_output = naive_blend(input_data)

    # --- [3. OUTPUT ASSERTIONS] ---
    roi_output = actual_output[input_dims[0]:input_dims[1], input_dims[2]:input_dims[3], :]
    np.testing.assert_allclose(roi_output, input_source, rtol=1e-5, err_msg="Full mask must copy source completely.")


def test_naive_blend_zero_mask():
    """
    [INPUT]:
        - Target: 10x10x3 constant array of 0.5.
        - Source: 4x4x3 constant array of 0.9.
        - Mask: 4x4x3 all 0.0 (0% alpha, 100% target preservation).
        - Dims: Bounding box ROI (1, 5, 1, 5).
    [OUTPUT]:
        - The entire output image must remain completely identical to the target.
    """
    # --- [1. INPUT] ---
    input_target = np.full((10, 10, 3), 0.5, dtype=np.float32)
    input_source = np.full((4, 4, 3), 0.9, dtype=np.float32)
    input_mask = np.zeros((4, 4, 3), dtype=np.float32)
    input_dims = (1, 5, 1, 5)

    input_data = {
        "target": input_target,
        "source": input_source,
        "mask": input_mask,
        "dims": input_dims,
    }

    # --- [2. ACT] ---
    actual_output = naive_blend(input_data)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(actual_output, input_target, rtol=1e-5, err_msg="Zero mask must preserve target completely.")


def test_naive_blend_half_mask():
    """
    [INPUT]:
        - Target: 8x8x3 constant array of 0.2.
        - Source: 4x4x3 constant array of 0.8.
        - Mask: 4x4x3 all 0.5 (50% linear alpha interpolation).
        - Dims: Bounding box ROI (2, 6, 2, 6).
    [OUTPUT]:
        - The blended ROI must equal 0.2 * (1 - 0.5) + 0.8 * 0.5 = 0.5 exactly.
    """
    # --- [1. INPUT] ---
    input_target = np.full((8, 8, 3), 0.2, dtype=np.float32)
    input_source = np.full((4, 4, 3), 0.8, dtype=np.float32)
    input_mask = np.full((4, 4, 3), 0.5, dtype=np.float32)
    input_dims = (2, 6, 2, 6)
    expected_roi_val = 0.5

    input_data = {
        "target": input_target,
        "source": input_source,
        "mask": input_mask,
        "dims": input_dims,
    }

    # --- [2. ACT] ---
    actual_output = naive_blend(input_data)

    # --- [3. OUTPUT ASSERTIONS] ---
    roi_output = actual_output[input_dims[0]:input_dims[1], input_dims[2]:input_dims[3], :]
    np.testing.assert_allclose(roi_output, expected_roi_val, rtol=1e-5, err_msg="50% mask must yield arithmetic mean.")


def test_naive_blend_preserves_unblended_regions(synthetic_blend_dict):
    """
    [INPUT]: Pre-configured synthetic dictionary from fixture with dims (4, 12, 4, 12).
    [OUTPUT]: All pixels outside the bounding box ROI must match the input target array.
    """
    # --- [1. INPUT] ---
    input_target = synthetic_blend_dict["target"]
    input_dims = synthetic_blend_dict["dims"]

    # --- [2. ACT] ---
    actual_output = naive_blend(synthetic_blend_dict)

    # --- [3. OUTPUT ASSERTIONS] ---
    # Top region outside ROI
    np.testing.assert_allclose(
        actual_output[:input_dims[0], :, :],
        input_target[:input_dims[0], :, :],
        err_msg="Top border outside ROI was altered.",
    )
    # Bottom region outside ROI
    np.testing.assert_allclose(
        actual_output[input_dims[1]:, :, :],
        input_target[input_dims[1]:, :, :],
        err_msg="Bottom border outside ROI was altered.",
    )


def test_naive_blend_immutable_input_target():
    """
    [INPUT]: Input dictionary with a target array.
    [OUTPUT]: The original input dictionary's target array must remain unmutated (copy semantics).
    """
    # --- [1. INPUT] ---
    original_target = np.zeros((10, 10, 3), dtype=np.float32)
    target_backup = original_target.copy()
    input_data = {
        "target": original_target,
        "source": np.ones((4, 4, 3), dtype=np.float32),
        "mask": np.ones((4, 4, 3), dtype=np.float32),
        "dims": (1, 5, 1, 5),
    }

    # --- [2. ACT] ---
    _ = naive_blend(input_data)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_array_equal(original_target, target_backup, err_msg="Input target array was mutated in-place.")
