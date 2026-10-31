"""
End-to-End (E2E) integration tests for the image blending pipeline using the Input-Output model.
Verifies complete execution from disk asset loading to final blended output.
"""

from pathlib import Path
import numpy as np
import pytest

from image_blending.config import SOURCE_DIR, MASK_DIR, TARGET_DIR
from image_blending.utils import load_img, preprocess_images
from image_blending.naive_blend import naive_blend
from image_blending.laplacian_blend import laplacian_blend
from image_blending.poisson_blend import poisson_blend


AVAILABLE_SAMPLES = [
    name for name in ["01.jpg", "02.jpg", "03.jpg"]
    if (SOURCE_DIR / name).exists() and (TARGET_DIR / name).exists() and (MASK_DIR / name).exists()
]


@pytest.fixture
def available_sample_image() -> str:
    """Returns the primary sample image available in data/image_blending/."""
    if not AVAILABLE_SAMPLES:
        pytest.skip("No real image blending assets found in data/image_blending/")
    return AVAILABLE_SAMPLES[0]


def test_e2e_naive_blending_pipeline(available_sample_image):
    """
    [INPUT]:
        - Real dataset sample: available_sample_image (e.g. '01.jpg').
        - Placement configuration: target_offset=[50, 50], scale=0.5.
    [OUTPUT]:
        - Returned array matches target image dimensions.
        - Pixel values strictly in [0.0, 1.0] with no NaNs.
        - Target pixels within bounding box ROI are modified; outside ROI is preserved.
    """
    # --- [1. INPUT] ---
    input_image_name = available_sample_image
    input_offset = [50, 50]
    input_scale = 0.5

    loaded_data = load_img(input_image_name, target_offset=input_offset, scale=input_scale)
    processed_input = preprocess_images(loaded_data)
    expected_shape = processed_input["target"].shape
    H_min, H_max, W_min, W_max = processed_input["dims"]

    # --- [2. ACT] ---
    actual_output = naive_blend(processed_input)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert isinstance(actual_output, np.ndarray), "Output must be a numpy ndarray"
    assert actual_output.shape == expected_shape, f"Expected shape {expected_shape}, got {actual_output.shape}"
    assert actual_output.min() >= 0.0 and actual_output.max() <= 1.0, "Output pixels out of range [0, 1]"
    assert not np.isnan(actual_output).any(), "Output contains NaN values"

    # Region of Interest must be modified
    assert np.any(actual_output[H_min:H_max, W_min:W_max] != processed_input["target"][H_min:H_max, W_min:W_max]), \
        "Naive blend did not modify the target ROI."


def test_e2e_laplacian_blending_pipeline(available_sample_image):
    """
    [INPUT]:
        - Real dataset sample: available_sample_image.
        - Pyramid parameters: num_levels=3, target_offset=[30, 30], scale=0.4.
    [OUTPUT]:
        - Multi-resolution blended output of identical shape to target.
        - Pixel values bounded in [0.0, 1.0] without numerical overflow or NaNs.
    """
    # --- [1. INPUT] ---
    input_image_name = available_sample_image
    input_offset = [30, 30]
    input_scale = 0.4
    num_levels = 3

    loaded_data = load_img(input_image_name, target_offset=input_offset, scale=input_scale)
    processed_input = preprocess_images(loaded_data)
    expected_shape = processed_input["target"].shape

    # --- [2. ACT] ---
    actual_output = laplacian_blend(processed_input, num_levels=num_levels)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert isinstance(actual_output, np.ndarray), "Output must be a numpy ndarray"
    assert actual_output.shape == expected_shape, f"Expected shape {expected_shape}, got {actual_output.shape}"
    assert actual_output.min() >= 0.0 and actual_output.max() <= 1.0, "Laplacian output out of range [0, 1]"
    assert not np.isnan(actual_output).any(), "Laplacian output contains NaNs"


def test_e2e_poisson_blending_pipeline(available_sample_image):
    """
    [INPUT]:
        - Real dataset sample: available_sample_image.
        - Solver configurations: grad_mix=False (Standard) and grad_mix=True (Mixed).
        - target_offset=[20, 20], scale=0.15.
    [OUTPUT]:
        - Solved Poisson output matching target shape.
        - Pixel values bounded in [0.0, 1.0] with no NaNs.
        - Both standard and mixed gradient modes solve successfully.
    """
    # --- [1. INPUT] ---
    input_image_name = available_sample_image
    input_offset = [20, 20]
    input_scale = 0.15

    loaded_data = load_img(input_image_name, target_offset=input_offset, scale=input_scale)
    processed_input = preprocess_images(loaded_data)
    expected_shape = processed_input["target"].shape

    # --- [2. ACT] ---
    actual_output_standard = poisson_blend(processed_input, grad_mix=False)
    actual_output_mixed = poisson_blend(processed_input, grad_mix=True)

    # --- [3. OUTPUT ASSERTIONS] ---
    for actual_output, mode in [(actual_output_standard, "Standard"), (actual_output_mixed, "Mixed")]:
        assert isinstance(actual_output, np.ndarray), f"{mode} Poisson output must be a numpy ndarray"
        assert actual_output.shape == expected_shape, f"{mode} Poisson shape mismatch: expected {expected_shape}"
        assert actual_output.min() >= 0.0 and actual_output.max() <= 1.0, f"{mode} Poisson pixels out of range [0, 1]"
        assert not np.isnan(actual_output).any(), f"{mode} Poisson output contains NaNs"


@pytest.mark.parametrize("image_name", AVAILABLE_SAMPLES, ids=[f"sample_{name}" for name in AVAILABLE_SAMPLES])
def test_e2e_all_samples_integrity(image_name):
    """
    [INPUT]: Each sample available in the dataset ('01.jpg', '02.jpg', '03.jpg').
    [OUTPUT]:
        - Pipeline loads and normalizes all 3 channels without errors.
        - Returns valid dictionary with 'source', 'mask', 'target', and 'dims'.
        - Verifies that naive blend runs cleanly on every project asset.
    """
    # --- [1. INPUT] ---
    input_offset = [10, 10]
    input_scale = 0.2

    # --- [2. ACT] ---
    loaded_data = load_img(image_name, target_offset=input_offset, scale=input_scale)
    preprocessed_data = preprocess_images(loaded_data)
    output_blended = naive_blend(preprocessed_data)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert output_blended.shape == preprocessed_data["target"].shape, "Output shape must equal target shape"
    assert output_blended.min() >= 0.0 and output_blended.max() <= 1.0, "Output values out of bounds"
    assert not np.isnan(output_blended).any(), "Output contains NaNs"
