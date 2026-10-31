"""
Feature detection and description.

Three extractors share one interface (`detect_and_describe`) so that the *same* stitching pipeline can be run
with each of them and the results compared fairly:
* sift: OpenCV SIFT, DoG scale-space extrema, gradient-orientation histograms, 128 floats (L2)
* orb: OpenCV ORB, oriented FAST corners on an image pyramid + rotated BRIEF, 256 bits (Hamming)
* harris: Harris corners + MOPS patches written from scratch in harris_mops.py, 64 floats (L2)


"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .harris_mops import detect_harris_mops

METHODS = ("sift", "orb", "harris")

DESCRIPTIONS = {
    "sift": "SIFT (OpenCV)",
    "orb": "ORB (OpenCV)",
    "harris": "Harris + MOPS (from scratch)",
}


@dataclass
class Features:
    """
    Keypoints and descriptors of one image.

    Attributes:
        method (str): The name of the feature extraction method used.
        kpts (np.ndarray): Array of shape (N, 2) containing float32 (x, y) coordinates in pixels.
            Pixel centres lie on integer coordinates (centre of the top-left pixel = (0, 0)),
            following the convention of warp.py and of the ground-truth homographies.
        desc (np.ndarray): Array of shape (N, D) containing descriptors. It is float32 for sift 
            and harris, or uint8 (8 bits per byte) for orb.
        scale (np.ndarray): Array of shape (N,) containing float32 keypoint diameter or window 
            size in pixels.
        response (np.ndarray): Array of shape (N,) containing float32 detector scores.
        binary (bool): True if Hamming distance should be used for descriptors, False for 
            Euclidean distance.
        seconds (float): Time taken for detection and description in seconds.
    """
    method: str
    kpts: np.ndarray        
    desc: np.ndarray        
    scale: np.ndarray       
    response: np.ndarray    
    binary: bool            
    seconds: float          

    def __len__(self) -> int:
        """Returns the number of keypoints detected."""
        return len(self.kpts)


ORB_SCALE_FACTOR, ORB_LEVELS = 1.2, 8


def _fix_opencv_coords(method: str, kps: list[cv2.KeyPoint], shape: tuple[int, int]) -> np.ndarray:
    """
    Corrects coordinates of OpenCV keypoints to follow the pixel-centre convention.

    The convention dictates that the centre of the top-left pixel is (0, 0).
    OpenCV does not follow that convention exactly, which biases the keypoints by a fraction of a pixel. 
    The bias is invisible as long as two views are processed identically, but it does not cancel when they are rotated or scaled
    against each other -- an exact 180-degree flip of an image makes it visible: mirrored keypoints should satisfy
    x' = (W - 1) - x, and for raw OpenCV coordinates they do not (see tests/test_panorama.py and the notebook).

    For SIFT: The first octave is the 2x up-sampled image (cv::resize, linear). Index u of the up-sampled image lies at
    u / 2 - 1/4 in the input (pixel-centre mapping x = (u + 1/2) / 2 - 1/2), but OpenCV reports u / 2. All
    later octaves are decimated from that base image, so every SIFT keypoint is shifted by +1/4 px: x = pt - 1/4.
    (OpenCV >= 4.8 offers SIFT_create(enable_precise_upscale=True) for the same purpose.)
    
    For ORB: Level l of the pyramid is the input resized (linear) to (W_l, H_l) = (round(W / s_l), round(H / s_l)),
    s_l = 1.2^l. Pixel-centre mapping: x = (x_l + 1/2) * W / W_l - 1/2, whereas OpenCV reports x_l * s_l.
    The pyramid sizes are rounded, so the true ratio W / W_l differs slightly from s_l as well.

    Args:
        method (str): The method used ("sift" or "orb").
        kps (list[cv2.KeyPoint]): The list of keypoints detected by OpenCV.
        shape (tuple[int, int]): The (height, width) of the original image.

    Returns:
        np.ndarray: Array of corrected coordinates of shape (N, 2).
    """
    pts = np.array([k.pt for k in kps], dtype=np.float64)
    if method == "sift":
        return pts - 0.25
    h, w = shape[:2]
    s = np.power(ORB_SCALE_FACTOR, np.array([k.octave for k in kps], dtype=np.float64))
    wl, hl = np.rint(w / s), np.rint(h / s)
    out = np.empty_like(pts)
    out[:, 0] = (pts[:, 0] / s + 0.5) * (w / wl) - 0.5
    out[:, 1] = (pts[:, 1] / s + 0.5) * (h / hl) - 0.5
    return out


def _from_opencv(method: str, kps: list[cv2.KeyPoint], desc: np.ndarray | None, binary: bool, seconds: float, shape: tuple[int, int, ...], fix_coords: bool = True) -> Features:
    """Converts OpenCV keypoints and descriptors into a Features object.

    Args:
        method (str): The method used ("sift" or "orb").
        kps (list[cv2.KeyPoint]): The list of keypoints detected by OpenCV.
        desc (np.ndarray | None): The corresponding descriptors.
        binary (bool): Indicates if the descriptors are binary (True for ORB, False for SIFT).
        seconds (float): Processing time in seconds.
        shape (tuple[int, int, ...]): Shape of the image.
        fix_coords (bool, optional): Whether to apply pixel-centre correction. Defaults to True.

    Returns:
        Features: A structured Features object containing the keypoints and descriptors.
    """
    if not kps:
        d = 32 if binary else 128
        return Features(method, np.zeros((0, 2), np.float32), np.zeros((0, d), np.uint8 if binary else np.float32),
                        np.zeros(0, np.float32), np.zeros(0, np.float32), binary, seconds)
    pts = _fix_opencv_coords(method, kps, shape) if fix_coords else np.array([k.pt for k in kps])
    return Features(
        method,
        pts.astype(np.float32),
        desc,
        np.array([k.size for k in kps], dtype=np.float32),
        np.array([k.response for k in kps], dtype=np.float32),
        binary,
        seconds,
    )


def detect_and_describe(
    gray: np.ndarray,
    method: str = "sift",
    max_keypoints: int = 4000,
    fix_coords: bool = True, 
    sift_contrast: float = 0.04, 
    fast_threshold: int = 20
) -> Features:
    """
    Detects up to `max_keypoints` keypoints in a grayscale image and computes their descriptors.

    Args:
        gray (np.ndarray): A single-channel uint8 grayscale image.
        method (str, optional): Feature extraction method to use ("sift", "orb", or "harris"). Defaults to "sift".
        max_keypoints (int, optional): Maximum number of keypoints to return. Defaults to 4000.
        fix_coords (bool, optional): If True, applies pixel-centre correction to OpenCV coordinates. Defaults to True.
        sift_contrast (float, optional): Absolute contrast threshold for SIFT (OpenCV default is 0.04). Lowering it helps on dark images. Defaults to 0.04.
        fast_threshold (int, optional): FAST corner threshold for ORB in grey levels. Defaults to 20.

    Returns:
        Features: Extracted keypoints and descriptors encapsulated in a Features object.

    Raises:
        ValueError: If `gray` is not a 2D single-channel image or if `method` is unknown.
    """
    method = method.lower()
    if gray.ndim != 2:
        raise ValueError("detect_and_describe expects a single-channel image")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    t0 = time.perf_counter()
    if method == "sift":
        kps, desc = cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=sift_contrast).detectAndCompute(gray, None)
        return _from_opencv(method, kps, desc, False, time.perf_counter() - t0, gray.shape, fix_coords)
    if method == "orb":
        orb = cv2.ORB_create(
            nfeatures=max_keypoints, scaleFactor=ORB_SCALE_FACTOR, nlevels=ORB_LEVELS,
            fastThreshold=int(fast_threshold)
        )
        kps, desc = orb.detectAndCompute(gray, None)
        return _from_opencv(method, kps, desc, True, time.perf_counter() - t0, gray.shape, fix_coords)
    if method == "harris":
        pts, desc, size, resp = detect_harris_mops(gray, max_keypoints=max_keypoints)
        return Features(method, pts, desc, size, resp, False, time.perf_counter() - t0)
    raise ValueError(f"Unknown feature method '{method}'. Choose from {METHODS}.")