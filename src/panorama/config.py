"""
Configuration Module for Panorama Stitching

This module centralizes the configuration settings for the panorama image stitching pipeline. 
It defines the directory paths for input data and output results, and provides a comprehensive 
dataclass (`StitchConfig`) that encapsulates all algorithmic parameters required for the stitching 
process. This includes settings for image preprocessing, feature extraction, feature matching, 
homography estimation (RANSAC), and image blending.

"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

#=================================================
# Path config
#=================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
PANO_ROOT = DATA_ROOT / "panorama"
PANO_INPUT_DIR = PANO_ROOT / "input"
PANO_RESULT_DIR  = PANO_ROOT / "result"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

#=================================================
# Stitching config
#=================================================

@dataclass
class StitchConfig:
    """Configuration parameters for the panorama stitching pipeline.

    Attributes:
        max_side (int | None, optional): The maximum length of the image's longer side. 
            Images will be resized so that max(H, W) <= max_side. If None, the original 
            size is kept. Defaults to 1000.
        method (str, optional): The feature extraction method to use. Options are 
            "sift", "orb", or "harris". Defaults to "sift".
        max_keypoints (int, optional): The maximum number of keypoints to extract. 
            Keeping this constant ensures a fair comparison across methods. Defaults to 4000.
        fix_coords (bool, optional): If True, applies a pixel-centre correction to 
            OpenCV's SIFT/ORB coordinates. Defaults to True.
        ratio (float, optional): Lowe's ratio test threshold for feature matching. 
            Defaults to 0.75.
        mutual (bool, optional): If True, requires matches to be mutual (cross-check). 
            Defaults to False.
        unique_matches (bool, optional): If True, enforces one-to-one matching. If multiple 
            features pick the same partner, only the most distinctive match is kept. 
            Defaults to True.
        ransac_thresh (float, optional): The RANSAC inlier threshold in pixels (relative 
            to the resized images). Defaults to 3.0.
        ransac_conf (float, optional): The desired probability of drawing at least one 
            outlier-free sample during RANSAC. Defaults to 0.995.
        ransac_max_iters (int, optional): The maximum number of RANSAC iterations. 
            Defaults to 3000.
        min_inliers (int, optional): The minimum number of RANSAC inliers required to 
            accept a pair of images as "overlapping". Defaults to 20.
        min_rank_ratio (float, optional): The minimum rank ratio (s_8 / s_1) of the DLT 
            matrix to uniquely determine the homography. Defaults to 1e-3.
        max_area_ratio (float, optional): The maximum allowed ratio between the mapped 
            image footprint and its original area to be considered a plausible mapping. 
            Defaults to 25.0.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        max_canvas_side (int, optional): The maximum side length of the final panorama 
            canvas. The canvas will be shrunk if it exceeds this value. Defaults to 6000.
        blend (str, optional): The blending method to use. Options are "overwrite", 
            "average", "feather", or "multiband". Defaults to "multiband".
        num_levels (int, optional): The number of pyramid levels to use if the multiband 
            blender is selected. Defaults to 5.
        gain_compensation (bool, optional): If True, equalizes the exposure of the images 
            before blending them. Defaults to False.
        crop (bool, optional): If True, crops the final output to the bounding box of 
            the valid pixels. Defaults to True.
    """
    # preprocessing
    max_side: int | None = 1000

    # feature extraction
    method: str = "sift"
    max_keypoints: int = 4000
    fix_coords: bool = True

    # matching
    ratio: float = 0.75
    mutual: bool = False
    unique_matches: bool = True

    # homography estimation
    ransac_thresh: float = 3.0
    ransac_conf: float = 0.995
    ransac_max_iters: int = 3000
    
    # acceptance criteria
    min_inliers: int = 20
    min_rank_ratio: float = 1e-3
    max_area_ratio: float = 25.0
    seed: int = 0

    # composition & blending
    max_canvas_side: int = 6000
    blend: str = "multiband"
    num_levels: int = 5
    gain_compensation: bool = False
    crop: bool = True