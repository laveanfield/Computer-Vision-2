"""
Unit tests for panorama.blend module following the Input-Output testing model.
"""

import numpy as np
import pytest
from panorama.blend import (
    compute_gains,
    blend_overwrite,
    blend_average,
    blend_feather,
    blend_multiband,
    _fill_invalid,
    blend_images,
    BLEND_MODES,
)


def test_compute_gains_equalizes_brightness():
    """
    [INPUT]:
        - Image 0: Canvas 20x40, left half valid [0:20, 0:25], constant 0.4.
        - Image 1: Canvas 20x40, right half valid [0:20, 15:40], constant 0.8.
        - Overlap is [0:20, 15:25] (10 pixels wide).
    [OUTPUT]: Gain for darker image 0 > 1.0, and gain for brighter image 1 < 1.0.
    """
    # --- [1. INPUT] ---
    im0 = np.full((20, 40, 3), 0.4, dtype=np.float32)
    m0 = np.zeros((20, 40), dtype=bool); m0[:, :25] = True

    im1 = np.full((20, 40, 3), 0.8, dtype=np.float32)
    m1 = np.zeros((20, 40), dtype=bool); m1[:, 15:] = True

    # --- [2. ACT] ---
    gains = compute_gains([im0, im1], [m0, m1])

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(gains) == 2
    assert gains[0] > 1.0, f"Expected gain > 1 for darker image, got {gains[0]}"
    assert gains[1] < 1.0, f"Expected gain < 1 for brighter image, got {gains[1]}"


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_blend_all_modes(mode):
    """
    [INPUT]: Two overlapping 32x32 RGB images on a 32x48 canvas; testing all BLEND_MODES.
    [OUTPUT]:
        - Output blended image has shape (32, 48, 3).
        - Coverage mask covers union of inputs.
        - All pixel values strictly within range [0.0, 1.0].
    """
    # --- [1. INPUT] ---
    im0 = np.full((32, 48, 3), 0.3, dtype=np.float32)
    m0 = np.zeros((32, 48), dtype=bool); m0[:, :32] = True

    im1 = np.full((32, 48, 3), 0.7, dtype=np.float32)
    m1 = np.zeros((32, 48), dtype=bool); m1[:, 16:] = True

    # --- [2. ACT] ---
    pano, cov, info = blend_images(
        [im0, im1], [m0, m1], mode=mode, num_levels=2, order=[0, 1], gain_compensation=False
    )

    # --- [3. OUTPUT ASSERTIONS] ---
    assert pano.shape == (32, 48, 3), f"Mode {mode} returned wrong shape"
    assert cov.shape == (32, 48)
    assert cov.sum() == (m0 | m1).sum(), "Coverage must equal union of masks"
    assert pano.min() >= 0.0 and pano.max() <= 1.0, f"Mode {mode} produced out-of-bounds pixels"
    assert not np.isnan(pano).any(), f"Mode {mode} produced NaNs"


def test_blend_single_image():
    """
    [INPUT]: Single image and mask passed to blend_images.
    [OUTPUT]: Output panorama exactly equals input image and mask.
    """
    # --- [1. INPUT] ---
    img = np.random.uniform(0.1, 0.9, size=(32, 32, 3)).astype(np.float32)
    mask = np.ones((32, 32), dtype=bool)

    # --- [2. ACT] ---
    pano, cov, _ = blend_images([img], [mask], mode="multiband", num_levels=2)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(pano, img, atol=1e-5)
    np.testing.assert_array_equal(cov, mask)


def test_blend_with_gain_compensation():
    """
    [INPUT]: Two overlapping images blended with gain_compensation=True.
    [OUTPUT]: Returns blended panorama and info dictionary containing 'gains'.
    """
    # --- [1. INPUT] ---
    im0 = np.full((32, 48, 3), 0.3, dtype=np.float32)
    m0 = np.zeros((32, 48), dtype=bool); m0[:, :32] = True
    im1 = np.full((32, 48, 3), 0.7, dtype=np.float32)
    m1 = np.zeros((32, 48), dtype=bool); m1[:, 16:] = True

    # --- [2. ACT] ---
    pano, cov, info = blend_images(
        [im0, im1], [m0, m1], mode="feather", gain_compensation=True
    )

    # --- [3. OUTPUT ASSERTIONS] ---
    assert "gains" in info
    assert len(info["gains"]) == 2
    assert pano.shape == (32, 48, 3)


def test_blend_unknown_mode():
    """
    [INPUT]: Invalid blend mode name 'magical_blend'.
    [OUTPUT]: Raises ValueError with available BLEND_MODES.
    """
    # --- [1. INPUT] ---
    im = np.zeros((10, 10, 3), dtype=np.float32)
    m = np.ones((10, 10), dtype=bool)

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    with pytest.raises(ValueError, match="Unknown blend mode"):
        blend_images([im], [m], mode="magical_blend")


def test_compute_gains_disjoint_island_image():
    """
    [INPUT]:
        - Image 0: Canvas 20x60, valid [0:20, 0:25].
        - Image 1: Canvas 20x60, valid [0:20, 15:40] (overlaps with Image 0).
        - Image 2: Canvas 20x60, valid [0:20, 50:60] (zero overlap with Images 0 and 1).
    [OUTPUT]: Gains for images 0 and 1 are calculated, while isolated Image 2 retains gain = 1.0 (A[k,k] == 0 branch).
    """
    # --- [1. INPUT] ---
    im0 = np.full((20, 60, 3), 0.4, dtype=np.float32)
    m0 = np.zeros((20, 60), dtype=bool); m0[:, :25] = True

    im1 = np.full((20, 60, 3), 0.8, dtype=np.float32)
    m1 = np.zeros((20, 60), dtype=bool); m1[:, 15:40] = True

    im2 = np.full((20, 60, 3), 0.5, dtype=np.float32)
    m2 = np.zeros((20, 60), dtype=bool); m2[:, 50:] = True  # Disjoint island

    # --- [2. ACT] ---
    gains = compute_gains([im0, im1, im2], [m0, m1, m2])

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(gains) == 3
    assert np.isclose(gains[2], 1.0, atol=1e-5), "Disjoint image must keep default gain 1.0"


def test_fill_invalid_empty_mask():
    """
    [INPUT]: 10x10 RGB image with an all-False mask (no valid pixels).
    [OUTPUT]: Returns the original image unmodified.
    """
    # --- [1. INPUT] ---
    img = np.ones((10, 10, 3), dtype=np.float32) * 0.5
    mask = np.zeros((10, 10), dtype=bool)

    # --- [2. ACT] ---
    result = _fill_invalid(img, mask, margin=5)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_array_equal(result, img)
