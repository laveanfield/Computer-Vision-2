"""
Unit tests for image_blending.config module following the Input-Output testing model.
"""

from pathlib import Path
from unittest.mock import patch
from image_blending import config


def test_path_configurations():
    """
    [INPUT]: None (reads predefined directory constants in config.py).
    [OUTPUT]: All path constants must be valid pathlib.Path instances pointing to expected locations.
    """
    # --- [1. INPUT] ---
    expected_paths = [
        config.PROJECT_ROOT,
        config.DATA_ROOT,
        config.BLEND_ROOT,
        config.SOURCE_DIR,
        config.MASK_DIR,
        config.TARGET_DIR,
        config.RESULT_DIR,
    ]

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    for path in expected_paths:
        assert isinstance(path, Path), f"Expected pathlib.Path instance, got {type(path)}"


def test_default_constants():
    """
    [INPUT]: Module-level defaults for blending parameters.
    [OUTPUT]:
        - GRAD_MIX is a boolean flag.
        - NUM_LEVELS is a positive integer > 0.
        - IMAGE_NAME is a non-empty string.
        - MANUAL_OFFSETS is a dictionary of predefined placement coordinates.
    """
    # --- [1. INPUT & 2. ACT] ---
    grad_mix = config.GRAD_MIX
    num_levels = config.NUM_LEVELS
    image_name = config.IMAGE_NAME
    manual_offsets = config.MANUAL_OFFSETS

    # --- [3. OUTPUT ASSERTIONS] ---
    assert isinstance(grad_mix, bool), f"Expected bool for GRAD_MIX, got {type(grad_mix)}"
    assert isinstance(num_levels, int) and num_levels > 0, f"Expected positive int, got {num_levels}"
    assert isinstance(image_name, str) and len(image_name) > 0, "Expected valid image name string"
    assert isinstance(manual_offsets, dict) and len(manual_offsets) > 0, "Expected non-empty offsets dict"


def test_get_offset_disabled():
    """
    [INPUT]:
        - Image name: "01.jpg"
        - Config state: USE_MANUAL_OFFSET = False
    [OUTPUT]:
        - Expected: (None, None) since manual offset lookup is disabled.
    """
    # --- [1. INPUT] ---
    input_image_name = "01.jpg"

    # --- [2. ACT] ---
    with patch.object(config, "USE_MANUAL_OFFSET", False):
        actual_offset, actual_scale = config.get_offset(input_image_name)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_offset is None, f"Expected None offset when disabled, got {actual_offset}"
    assert actual_scale is None, f"Expected None scale when disabled, got {actual_scale}"


def test_get_offset_enabled_existing_image():
    """
    [INPUT]:
        - Image name: "01.jpg" (present in MANUAL_OFFSETS)
        - Config state: USE_MANUAL_OFFSET = True
    [OUTPUT]:
        - Expected offset: [200, 21]
        - Expected scale: 1.0
    """
    # --- [1. INPUT] ---
    input_image_name = "01.jpg"
    expected_offset = [200, 21]
    expected_scale = 1.0

    # --- [2. ACT] ---
    with patch.object(config, "USE_MANUAL_OFFSET", True):
        actual_offset, actual_scale = config.get_offset(input_image_name)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_offset == expected_offset, f"Expected offset {expected_offset}, got {actual_offset}"
    assert actual_scale == expected_scale, f"Expected scale {expected_scale}, got {actual_scale}"


def test_get_offset_enabled_missing_image():
    """
    [INPUT]:
        - Image name: "non_existent_image.jpg" (not present in MANUAL_OFFSETS)
        - Config state: USE_MANUAL_OFFSET = True
    [OUTPUT]:
        - Expected: (None, None) since the key is not in MANUAL_OFFSETS.
    """
    # --- [1. INPUT] ---
    input_image_name = "non_existent_image.jpg"

    # --- [2. ACT] ---
    with patch.object(config, "USE_MANUAL_OFFSET", True):
        actual_offset, actual_scale = config.get_offset(input_image_name)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_offset is None, f"Expected None for missing image offset, got {actual_offset}"
    assert actual_scale is None, f"Expected None for missing image scale, got {actual_scale}"
