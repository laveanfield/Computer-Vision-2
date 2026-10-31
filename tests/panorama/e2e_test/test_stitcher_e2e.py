"""
End-to-End (E2E) integration tests for the Panorama Stitching pipeline.
Verifies complete flow from image file loading to final blended panorama using the Input-Output model.
"""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from panorama.config import StitchConfig, PANO_INPUT_DIR
from panorama.stitcher import (
    PanoramaStitcher,
    PanoramaResult,
    list_images,
    load_images,
    preprocess,
    _hull_fraction,
    _max_spanning_forest,
)


def test_list_and_load_images(sample_mountain_paths):
    """
    [INPUT]: Directory path containing image files.
    [OUTPUT]:
        - list_images returns sorted Paths in natural order.
        - load_images loads all files into valid BGR numpy arrays.
    """
    # --- [1. INPUT] ---
    folder = PANO_INPUT_DIR / "mountain"

    # --- [2. ACT] ---
    paths = list_images(folder)
    images = load_images(paths)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(paths) >= 2
    assert len(images) == len(paths)
    for img in images:
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3 and img.shape[2] == 3


def test_stitch_insufficient_images():
    """
    [INPUT]: List containing fewer than 2 images (e.g. 1 image).
    [OUTPUT]: Returns PanoramaResult with success=False and message='Need at least two images'.
    """
    # --- [1. INPUT] ---
    single_image = [np.zeros((50, 50, 3), dtype=np.uint8)]
    stitcher = PanoramaStitcher()

    # --- [2. ACT] ---
    result = stitcher.stitch(single_image)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert isinstance(result, PanoramaResult)
    assert result.ok is False
    assert "Need at least two images" in result.message


def test_stitch_disconnected_unrelated_images():
    """
    [INPUT]: Two completely random, unrelated noise images with zero feature correspondence.
    [OUTPUT]: Returns PanoramaResult with ok=False (no pair with >= min_inliers).
    """
    # --- [1. INPUT] ---
    np.random.seed(123)
    im1 = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    im2 = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    stitcher = PanoramaStitcher(min_inliers=20)

    # --- [2. ACT] ---
    result = stitcher.stitch([im1, im2])

    # --- [3. OUTPUT ASSERTIONS] ---
    assert result.ok is False
    assert "No pair of images has enough consistent matches" in result.message


def test_stitch_e2e_mountain_scene(sample_mountain_paths):
    """
    [INPUT]:
        - Real dataset: 'mountain' scene (2 overlapping photographs).
        - StitchConfig: method='sift', blend='multiband', max_side=600 for fast CI testing.
    [OUTPUT]:
        - result.ok is True.
        - Exactly 2 images used (used=[0, 1]).
        - Stitched panorama is non-empty uint8 RGB image with valid coverage.
    """
    # --- [1. INPUT] ---
    images = load_images(sample_mountain_paths)
    cfg = StitchConfig(method="sift", blend="multiband", max_side=600, min_inliers=15)
    stitcher = PanoramaStitcher(cfg)

    # --- [2. ACT] ---
    result = stitcher.stitch(images, verbose=False)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert result.ok is True, f"Stitching failed: {result.message}"
    assert len(result.used) == 2, f"Expected 2 used images, got {result.used}"
    assert result.panorama is not None
    assert result.panorama.dtype == np.uint8
    assert result.panorama.shape[0] > 0 and result.panorama.shape[1] > 0
    assert result.coverage.sum() > 0


def test_stitch_e2e_aqueduct_scene(sample_aqueduct_paths):
    """
    [INPUT]:
        - Real dataset: 'aqueduct' scene (2 photographs).
        - StitchConfig: method='sift', blend='feather', max_side=500.
    [OUTPUT]:
        - result.ok is True.
        - Valid stitched output image and non-empty timing profile.
    """
    # --- [1. INPUT] ---
    images = load_images(sample_aqueduct_paths)
    cfg = StitchConfig(method="sift", blend="feather", max_side=500, min_inliers=15)
    stitcher = PanoramaStitcher(cfg)

    # --- [2. ACT] ---
    result = stitcher.stitch(images, verbose=False)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert result.ok is True, f"Stitching failed: {result.message}"
    assert len(result.used) == 2
    assert result.panorama.shape[0] > 0 and result.panorama.shape[1] > 0
    assert "features" in result.timings and "total" in result.timings


def test_stitch_e2e_orb_extractor(sample_mountain_paths):
    """
    [INPUT]:
        - 'mountain' scene stitched using binary ORB feature extractor.
    [OUTPUT]:
        - result.ok is True, proving cross-extractor compatibility.
    """
    # --- [1. INPUT] ---
    images = load_images(sample_mountain_paths)
    cfg = StitchConfig(method="orb", blend="average", max_side=600, min_inliers=15)
    stitcher = PanoramaStitcher(cfg)

    # --- [2. ACT] ---
    result = stitcher.stitch(images, verbose=False)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert result.ok is True, f"ORB stitching failed: {result.message}"
    assert result.panorama.shape[0] > 0


def test_load_images_missing_file_raises_filenotfound():
    """
    [INPUT]: List containing file path where image reading fails (returns None).
    [OUTPUT]: Raises FileNotFoundError indicating unreadable image.
    """
    missing_path = [Path("data/panorama/input/mountain/nonexistent_file_abc123.jpg")]
    with patch("cv2.imread", return_value=None):
        with pytest.raises(FileNotFoundError, match="Cannot read image"):
            load_images(missing_path)


def test_preprocess_grayscale_conversion():
    """
    [INPUT]: 2D single-channel grayscale image (50, 50).
    [OUTPUT]: preprocess converts it to a 3-channel (50, 50, 3) BGR array.
    """
    # --- [1. INPUT] ---
    gray_img = np.ones((50, 50), dtype=np.uint8) * 128

    # --- [2. ACT] ---
    processed = preprocess([gray_img], max_side=None)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert len(processed) == 1
    assert processed[0].ndim == 3 and processed[0].shape[2] == 3


def test_hull_fraction_insufficient_points():
    """
    [INPUT]: 2 points (< 3 points needed for convex hull area).
    [OUTPUT]: _hull_fraction returns 0.0.
    """
    pts = np.array([[10.0, 10.0], [20.0, 20.0]])
    frac = _hull_fraction(pts, (100, 100))
    assert frac == 0.0


def test_stitch_crop_false(sample_mountain_paths):
    """
    [INPUT]: Mountain images stitched with cfg.crop = False.
    [OUTPUT]: Returns successful stitched panorama without auto-cropping bounds.
    """
    # --- [1. INPUT] ---
    images = load_images(sample_mountain_paths)
    cfg = StitchConfig(method="sift", blend="feather", max_side=500, crop=False, min_inliers=15)
    stitcher = PanoramaStitcher(cfg)

    # --- [2. ACT] ---
    result = stitcher.stitch(images, verbose=False)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert result.ok is True
    assert result.panorama is not None
    assert result.coverage is not None


def test_max_spanning_forest_path_compression_and_cycle():
    """
    [INPUT]: Graph with 4 nodes: (0, 1), (1, 2), (2, 3), and cycle-closing edge (0, 3).
    [OUTPUT]: _max_spanning_forest compresses multi-hop paths (lines 239-240) and rejects cycle (branch 246->244).
    """
    edges = [(0, 1, 100), (1, 2, 90), (2, 3, 80), (0, 3, 50)]
    tree = _max_spanning_forest(4, edges)
    assert len(tree) == 3
    assert (0, 3, 50) not in tree


def test_stitch_degenerate_exploding_canvas(sample_mountain_paths):
    """
    [INPUT]: compute_canvas returning massive dimensions (50000, 50000).
    [OUTPUT]: stitch detects area ratio > MAX_AREA_RATIO, raising ValueError and returning PanoramaResult(ok=False) (lines 468-472).
    """
    images = load_images(sample_mountain_paths)
    stitcher = PanoramaStitcher(StitchConfig(method="sift", max_side=400, min_inliers=15))
    with patch("panorama.stitcher.compute_canvas", return_value=((50000, 50000), np.eye(3))):
        result = stitcher.stitch(images, verbose=False)
        assert result.ok is False
        assert "is more than 40x the input area" in result.message


def test_stitch_small_overlap_skips_metric_recording(sample_mountain_paths):
    """
    [INPUT]: overlap_photometric_error mocked to return n_pixels = 100 (< 500 threshold).
    [OUTPUT]: Pair is skipped from result.overlap dictionary (branch 492->490).
    """
    images = load_images(sample_mountain_paths)
    stitcher = PanoramaStitcher(StitchConfig(method="sift", max_side=400, min_inliers=15))
    with patch("panorama.stitcher.overlap_photometric_error", return_value={"n_pixels": 100, "mae": 0.0, "ncc": 1.0}):
        result = stitcher.stitch(images, verbose=False)
        assert result.ok is True
        assert len(result.overlap) == 0
