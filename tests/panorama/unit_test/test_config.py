"""
Unit tests for panorama.config module following the Input-Output testing model.
"""

from pathlib import Path
from panorama import config
from panorama.config import StitchConfig


def test_path_configurations():
    """
    [INPUT]: Predefined directory constants in panorama.config.
    [OUTPUT]: All path variables must be valid pathlib.Path instances pointing to expected locations.
    """
    # --- [1. INPUT] ---
    expected_paths = [
        config.PROJECT_ROOT,
        config.DATA_ROOT,
        config.PANO_ROOT,
        config.PANO_INPUT_DIR,
        config.PANO_RESULT_DIR,
    ]

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    for path in expected_paths:
        assert isinstance(path, Path), f"Expected pathlib.Path instance, got {type(path)}"
    assert config.PANO_ROOT == config.DATA_ROOT / "panorama"


def test_image_extensions():
    """
    [INPUT]: Supported image file extensions tuple in panorama.config.
    [OUTPUT]: Must include standard extensions ('.jpg', '.png', '.jpeg').
    """
    # --- [1. INPUT & 2. ACT] ---
    exts = config.IMAGE_EXTENSIONS

    # --- [3. OUTPUT ASSERTIONS] ---
    assert isinstance(exts, tuple), "IMAGE_EXTENSIONS must be a tuple"
    assert ".jpg" in exts and ".png" in exts and ".jpeg" in exts


def test_stitch_config_defaults():
    """
    [INPUT]: Default initialization of StitchConfig().
    [OUTPUT]: Default algorithmic parameters must match specification (SIFT, ratio 0.75, multiband).
    """
    # --- [1. INPUT & 2. ACT] ---
    cfg = StitchConfig()

    # --- [3. OUTPUT ASSERTIONS] ---
    assert cfg.method == "sift"
    assert cfg.max_side == 1000
    assert cfg.max_keypoints == 4000
    assert cfg.fix_coords is True
    assert cfg.ratio == 0.75
    assert cfg.mutual is False
    assert cfg.unique_matches is True
    assert cfg.ransac_thresh == 3.0
    assert cfg.min_inliers == 20
    assert cfg.blend == "multiband"
    assert cfg.num_levels == 5
    assert cfg.crop is True


def test_stitch_config_custom_overrides():
    """
    [INPUT]: Custom parameters passed to StitchConfig (method='orb', blend='feather', max_side=600).
    [OUTPUT]: Instantiated config must reflect the overridden fields while preserving defaults for others.
    """
    # --- [1. INPUT & 2. ACT] ---
    cfg = StitchConfig(method="orb", blend="feather", max_side=600, min_inliers=15)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert cfg.method == "orb"
    assert cfg.blend == "feather"
    assert cfg.max_side == 600
    assert cfg.min_inliers == 15
    assert cfg.ratio == 0.75  # Preserved default
