"""
Unit tests for image_blending.utils module following the Input-Output testing model.
"""

from unittest.mock import patch
import cv2
import numpy as np
import pytest
from image_blending.utils import (
    load_img,
    display_images,
    preprocess_images,
    select_offset,
)


def test_load_img_source_not_found():
    """
    [INPUT]: Filename "missing_image.jpg" where source image fails to load.
    [OUTPUT]: Raises FileNotFoundError with meaningful context matching "Source not found".
    """
    # --- [1. INPUT] ---
    input_filename = "missing_image.jpg"

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    with patch("cv2.imread", return_value=None):
        with pytest.raises(FileNotFoundError, match="Source not found"):
            load_img(input_filename, target_offset=[0, 0])


def test_load_img_mask_not_found():
    """
    [INPUT]: Source exists, but mask image fails to load (None).
    [OUTPUT]: Raises FileNotFoundError matching "Mask not found".
    """
    # --- [1. INPUT] ---
    def mock_read(path_str):
        if "source" in path_str:
            return np.ones((10, 10, 3), dtype=np.uint8)
        return None

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    with patch("cv2.imread", side_effect=mock_read):
        with pytest.raises(FileNotFoundError, match="Mask not found"):
            load_img("missing_mask.jpg", target_offset=[0, 0])


def test_load_img_target_not_found():
    """
    [INPUT]: Source and mask exist, but target background fails to load (None).
    [OUTPUT]: Raises FileNotFoundError matching "Target not found".
    """
    # --- [1. INPUT] ---
    def mock_read(path_str):
        if "source" in path_str or "mask" in path_str:
            return np.ones((10, 10, 3), dtype=np.uint8)
        return None

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    with patch("cv2.imread", side_effect=mock_read):
        with pytest.raises(FileNotFoundError, match="Target not found"):
            load_img("missing_target.jpg", target_offset=[0, 0])


def test_load_img_normalization_and_scaling():
    """
    [INPUT]:
        - Source (20x20 uint8), Mask (20x20 uint8), Target (40x40 uint8).
        - Requested scale: 0.5, target_offset: [10, 10].
    [OUTPUT]:
        - Loaded images normalized to float range [0.0, 1.0].
        - Source and mask scaled by 0.5 to (10, 10).
        - 'dims' assigned to [10, 10].
    """
    # --- [1. INPUT] ---
    raw_source = np.full((20, 20, 3), 128, dtype=np.uint8)
    raw_mask = np.full((20, 20, 3), 255, dtype=np.uint8)
    raw_target = np.full((40, 40, 3), 64, dtype=np.uint8)
    input_scale = 0.5
    input_offset = [10, 10]

    def mock_read(path_str):
        if "source" in path_str: return raw_source
        if "mask" in path_str: return raw_mask
        if "target" in path_str: return raw_target
        return None

    # --- [2. ACT] ---
    with patch("cv2.imread", side_effect=mock_read):
        actual_output = load_img("01.jpg", target_offset=input_offset, scale=input_scale)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_output["source"].max() <= 1.0, "Source was not normalized to [0, 1]"
    assert actual_output["mask"].max() <= 1.0, "Mask was not normalized to [0, 1]"
    assert actual_output["target"].max() <= 1.0, "Target was not normalized to [0, 1]"
    assert actual_output["source"].shape[:2] == (10, 10), f"Expected scaled shape (10, 10), got {actual_output['source'].shape[:2]}"
    assert actual_output["dims"] == input_offset, "Offset dims mismatch"


def test_load_img_interactive_fallback(mock_cv2_gui):
    """
    [INPUT]: target_offset = None (invokes interactive offset selection).
    [OUTPUT]: select_offset is invoked and its returned offset [5, 5] is set as 'dims'.
    """
    # --- [1. INPUT] ---
    dummy_img = np.ones((10, 10, 3), dtype=np.uint8)
    expected_offset = [5, 5]

    # --- [2. ACT] ---
    with patch("cv2.imread", return_value=dummy_img), \
         patch("image_blending.utils.select_offset", return_value=(expected_offset, 1.0)) as mock_select:
        actual_output = load_img("01.jpg", target_offset=None)

    # --- [3. OUTPUT ASSERTIONS] ---
    mock_select.assert_called_once_with("01.jpg")
    assert actual_output["dims"] == expected_offset, "Interactive offset was not propagated to data dict."


def test_preprocess_images_standard_offset():
    """
    [INPUT]: Source 10x10, Target 30x30, offset [5, 5] (fully contained).
    [OUTPUT]: Bounding box dims = (5, 15, 5, 15); source remains uncropped at (10, 10, 3).
    """
    # --- [1. INPUT] ---
    input_data = {
        "source": np.ones((10, 10, 3), dtype=np.float32),
        "mask": np.ones((10, 10, 3), dtype=np.float32),
        "target": np.zeros((30, 30, 3), dtype=np.float32),
        "dims": [5, 5],
    }

    # --- [2. ACT] ---
    actual_output = preprocess_images(input_data)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_output["dims"] == (5, 15, 5, 15), f"Expected (5, 15, 5, 15), got {actual_output['dims']}"
    assert actual_output["source"].shape == (10, 10, 3), "Fully contained source should not be cropped."


def test_preprocess_images_crop_exceeding_bounds():
    """
    [INPUT]: Source 20x20 placed at [10, 10] on Target 25x25 (extends past target boundary by 5 pixels).
    [OUTPUT]: Source and mask cropped to 15x15; bounding box clamped to (10, 25, 10, 25).
    """
    # --- [1. INPUT] ---
    input_data = {
        "source": np.ones((20, 20, 3), dtype=np.float32),
        "mask": np.ones((20, 20, 3), dtype=np.float32),
        "target": np.zeros((25, 25, 3), dtype=np.float32),
        "dims": [10, 10],
    }

    # --- [2. ACT] ---
    actual_output = preprocess_images(input_data)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_output["dims"] == (10, 25, 10, 25), "Bounding box failed to clamp at target boundary."
    assert actual_output["source"].shape == (15, 15, 3), f"Expected cropped shape (15, 15, 3), got {actual_output['source'].shape}"
    assert actual_output["mask"].shape == (15, 15, 3)


def test_preprocess_images_negative_offset():
    """
    [INPUT]: Source 10x10, Target 30x30, offset [-3, -2] (negative coordinates).
    [OUTPUT]:
        - Offsets clamped: H_min=0, W_min=0, H_max=10, W_max=10.
        - Rolled edges zeroed out: vertical indices 7: and horizontal indices 8: must be zeros.
    """
    # --- [1. INPUT] ---
    input_data = {
        "source": np.ones((10, 10, 3), dtype=np.float32),
        "mask": np.ones((10, 10, 3), dtype=np.float32),
        "target": np.zeros((30, 30, 3), dtype=np.float32),
        "dims": [-3, -2],
    }

    # --- [2. ACT] ---
    actual_output = preprocess_images(input_data)

    # --- [3. OUTPUT ASSERTIONS] ---
    H_min, H_max, W_min, W_max = actual_output["dims"]
    assert H_min == 0 and W_min == 0, "Negative offset coordinates must be clamped to 0."
    assert H_max == 10 and W_max == 10, "Bounding box upper bounds incorrect."

    # Verify rolled border zeroing
    assert np.all(actual_output["source"][7:, :, :] == 0), "Rolled vertical margin was not zeroed out."
    assert np.all(actual_output["mask"][7:, :, :] == 0)
    assert np.all(actual_output["source"][:, 8:, :] == 0), "Rolled horizontal margin was not zeroed out."
    assert np.all(actual_output["mask"][:, 8:, :] == 0)


def test_display_images_headless(mock_plt_show):
    """
    [INPUT]: Image data dictionary with source, mask, target; with and without title string.
    [OUTPUT]: Matplotlib display functions execute without raising errors; plt.show called twice.
    """
    # --- [1. INPUT] ---
    input_data = {
        "source": np.ones((8, 8, 3), dtype=np.float32),
        "mask": np.zeros((8, 8, 3), dtype=np.float32),
        "target": np.zeros((16, 16, 3), dtype=np.float32),
        "dims": [0, 0],
    }

    # --- [2. ACT] ---
    display_images(input_data, title="With Title")
    display_images(input_data, title="")

    # --- [3. OUTPUT ASSERTIONS] ---
    assert mock_plt_show.call_count == 2, f"Expected 2 plt.show invocations, got {mock_plt_show.call_count}"


def test_select_offset_interactive_events():
    """
    [INPUT]:
        - Simulated OpenCV event stream:
          * Left click at (2, 2) and (25, 25).
          * Mouse move at (4, 4).
          * Mouse wheel zoom (+1 and -1).
          * Key sequence: [space (ignored), '+' (+0.05), '=' (+0.05), '-' (-0.05), Enter (13)].
    [OUTPUT]:
        - Offset confirmed at [25, 25].
        - Scale confirmed at 1.05x.
    """
    # --- [1. INPUT] ---
    dummy_img = np.zeros((20, 20, 3), dtype=np.uint8)
    key_sequence = [ord(' '), ord('+'), ord('='), ord('-'), 13]
    key_iter = iter(key_sequence)

    def driver(win_name, callback):
        callback(cv2.EVENT_LBUTTONDOWN, 2, 2, 0, None)
        callback(cv2.EVENT_MOUSEMOVE, 4, 4, 0, None)
        callback(cv2.EVENT_LBUTTONDOWN, 25, 25, 0, None)  # triggers else branch
        callback(cv2.EVENT_MOUSEWHEEL, 0, 0, 1, None)     # scale up
        callback(cv2.EVENT_MOUSEWHEEL, 0, 0, -1, None)    # scale down
        callback(cv2.EVENT_RBUTTONDOWN, 0, 0, 0, None)    # unhandled event

    # --- [2. ACT] ---
    with patch("cv2.imread", return_value=dummy_img), \
         patch("cv2.imshow") as mock_imshow, \
         patch("cv2.setMouseCallback", side_effect=driver), \
         patch("cv2.waitKey", side_effect=lambda delay: next(key_iter)), \
         patch("cv2.destroyAllWindows") as mock_destroy:
        actual_offset, actual_scale = select_offset("01.jpg")

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_offset == [25, 25], f"Expected offset [25, 25], got {actual_offset}"
    assert np.isclose(actual_scale, 1.05), f"Expected scale 1.05, got {actual_scale}"
    mock_imshow.assert_called()
    mock_destroy.assert_called_once()
