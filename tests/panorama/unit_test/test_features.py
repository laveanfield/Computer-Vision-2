"""
Unit tests for panorama.features module following the Input-Output testing model.
"""

import numpy as np
import pytest
from panorama.features import detect_and_describe, Features, METHODS


@pytest.mark.parametrize("method", METHODS)
def test_detect_and_describe_all_methods(synthetic_corner_image, method):
    """
    [INPUT]: synthetic_corner_image, testing all supported methods ('sift', 'orb', 'harris').
    [OUTPUT]:
        - Returns a valid Features instance.
        - Number of keypoints > 0 and <= max_keypoints (100).
        - Descriptors shape matches (N, D) with correct float/binary types.
    """
    # --- [1. INPUT] ---
    img_gray = synthetic_corner_image
    max_kpts = 100

    # --- [2. ACT] ---
    feats = detect_and_describe(img_gray, method=method, max_keypoints=max_kpts, fix_coords=True)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert isinstance(feats, Features)
    assert len(feats) > 0 and len(feats) <= max_kpts
    assert feats.kpts.shape == (len(feats), 2)
    assert feats.method == method

    if method == "sift":
        assert feats.desc.shape == (len(feats), 128)
        assert feats.binary is False
    elif method == "orb":
        assert feats.desc.shape == (len(feats), 32)
        assert feats.binary is True
    elif method == "harris":
        assert feats.desc.shape == (len(feats), 64)
        assert feats.binary is False


def test_detect_and_describe_unknown_method():
    """
    [INPUT]: method = 'invalid_detector_xyz'.
    [OUTPUT]: Raises ValueError detailing available methods.
    """
    # --- [1. INPUT] ---
    img_dummy = np.zeros((50, 50), dtype=np.uint8)

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    with pytest.raises(ValueError, match="Unknown feature method"):
        detect_and_describe(img_dummy, method="invalid_detector_xyz")


def test_fix_coords_flag(synthetic_corner_image):
    """
    [INPUT]: SIFT feature extraction with fix_coords=True vs fix_coords=False.
    [OUTPUT]: Keypoint coordinates with fix_coords=False should differ from fix_coords=True by 0.25px.
    """
    # --- [1. INPUT] ---
    img_gray = synthetic_corner_image

    # --- [2. ACT] ---
    feats_fixed = detect_and_describe(img_gray, method="sift", max_keypoints=50, fix_coords=True)
    feats_raw = detect_and_describe(img_gray, method="sift", max_keypoints=50, fix_coords=False)

    # --- [3. OUTPUT ASSERTIONS] ---
    if len(feats_fixed) > 0 and len(feats_raw) > 0:
        # Check coordinate shift (-0.25) applied in fix_coords for SIFT
        diff = feats_raw.kpts[0] - feats_fixed.kpts[0]
        np.testing.assert_allclose(diff, [0.25, 0.25], atol=1e-3)


def test_detect_and_describe_3d_input_raises_value_error():
    """
    [INPUT]: 3D RGB image array of shape (50, 50, 3).
    [OUTPUT]: Raises ValueError indicating single-channel image requirement.
    """
    # --- [1. INPUT] ---
    img_3d = np.zeros((50, 50, 3), dtype=np.uint8)

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    with pytest.raises(ValueError, match="single-channel"):
        detect_and_describe(img_3d, method="sift")


def test_detect_and_describe_float_dtype_conversion(synthetic_corner_image):
    """
    [INPUT]: Grayscale image in float32 format [0.0, 255.0].
    [OUTPUT]: detect_and_describe handles type conversion to uint8 and returns valid features.
    """
    # --- [1. INPUT] ---
    img_float = synthetic_corner_image.astype(np.float32)

    # --- [2. ACT] ---
    feats = detect_and_describe(img_float, method="sift", max_keypoints=30)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert isinstance(feats, Features)
    assert len(feats) > 0


def test_detect_and_describe_zero_keypoints_empty_features():
    """
    [INPUT]: Uniform black 30x30 image with no texture.
    [OUTPUT]: Returns empty Features object with shape (0, 2) keypoints and (0, D) descriptors.
    """
    # --- [1. INPUT] ---
    flat_img = np.zeros((30, 30), dtype=np.uint8)

    # --- [2. ACT] ---
    feats_sift = detect_and_describe(flat_img, method="sift")
    feats_orb = detect_and_describe(flat_img, method="orb")

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(feats_sift) == 0
    assert feats_sift.kpts.shape == (0, 2)
    assert feats_sift.desc.shape == (0, 128)
    assert len(feats_orb) == 0
    assert feats_orb.kpts.shape == (0, 2)
    assert feats_orb.desc.shape == (0, 32)
