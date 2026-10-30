"""
Quality measures for the stitching experiments.

Functions:
  * overlap_photometric_error: Measures how well two *warped* images agree inside their overlap. 
    This needs no ground truth and reacts to misalignment (ghosting) as well as exposure differences.
  * overlap_ncc_bands: Measures the same agreement resolved in horizontal bands to show *where* 
    two images disagree.
  * homography_rms_error: Measures the geometric accuracy against a known ground-truth homography 
    (used in synthetic tests).
"""
from __future__ import annotations

import cv2
import numpy as np

from .homography import apply_homography


def _luminance(img: np.ndarray) -> np.ndarray:
    """Converts a BGR float image to a luminance (grayscale) image.

    Args:
        img (np.ndarray): The input BGR image array with values in [0, 1].

    Returns:
        np.ndarray: The resulting luminance image array.
    """
    return img @ np.array([0.114, 0.587, 0.299], dtype=np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the zero-mean normalized cross-correlation (NCC) between two arrays.

    Args:
        a (np.ndarray): The first input array.
        b (np.ndarray): The second input array.

    Returns:
        float: The NCC value, or NaN if the standard deviation of either array is too close to zero.
    """
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def overlap_photometric_error(warped_a: np.ndarray, mask_a: np.ndarray, warped_b: np.ndarray, mask_b: np.ndarray,
                              erode: int = 9, min_pixels: int = 500) -> dict:
    """Compares two warped images over their overlapping footprints.

    The overlap region is eroded by `erode` pixels to avoid interpolation artifacts at the footprint borders.

    Metrics returned:
      - mae: Mean absolute luminance difference in grey levels (0-255). Sensitive to exposure differences.
      - ncc: Zero-mean normalized cross-correlation of luminance. Invariant to gain and bias.
      - ncc_hp: NCC after applying a high-pass filter (Difference of Gaussians). Reacts strongly to ghosting/misalignment.

    Args:
        warped_a (np.ndarray): The first warped image (canvas size, float32, BGR in [0, 1]).
        mask_a (np.ndarray): Boolean mask indicating valid pixels for `warped_a`.
        warped_b (np.ndarray): The second warped image (canvas size, float32, BGR in [0, 1]).
        mask_b (np.ndarray): Boolean mask indicating valid pixels for `warped_b`.
        erode (int, optional): The number of pixels to erode the overlap boundary by. Defaults to 9.
        min_pixels (int, optional): Minimum required number of overlapping pixels to compute metrics. Defaults to 500.

    Returns:
        dict: A dictionary containing the metrics 'n_pixels', 'mae', 'ncc', and 'ncc_hp'. 
            If the overlap area is too small, metrics will be NaN.
    """
    ov = mask_a & mask_b
    if erode > 0:
        ov = cv2.erode(ov.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)).astype(bool)
    n = int(ov.sum())
    out = {"n_pixels": n, "mae": float("nan"), "ncc": float("nan"), "ncc_hp": float("nan")}
    if n < min_pixels:
        return out

    ga, gb = _luminance(warped_a), _luminance(warped_b)
    out["mae"] = float(np.abs(ga[ov] - gb[ov]).mean() * 255.0)
    out["ncc"] = _ncc(ga[ov], gb[ov])
    hp = lambda g: g - cv2.GaussianBlur(g, (0, 0), 3.0)
    out["ncc_hp"] = _ncc(hp(ga)[ov], hp(gb)[ov])
    return out


def overlap_ncc_bands(warped_a: np.ndarray, mask_a: np.ndarray, warped_b: np.ndarray, mask_b: np.ndarray,
                      n_bands: int = 6, erode: int = 9, min_pixels: int = 500) -> list[float]:
    """Computes high-pass NCC within horizontal bands of the eroded overlap region.

    This shows *where* two warped images agree spatially, from top to bottom.
    A well-registered static structure yields values close to 1. Ghosting, moving water, 
    or moving foliage lower the value. An almost uniform region (e.g., clear sky) has no 
    structure to correlate, so its value says nothing about the registration accuracy.

    Args:
        warped_a (np.ndarray): The first warped image (canvas size, float32, BGR in [0, 1]).
        mask_a (np.ndarray): Boolean mask indicating valid pixels for `warped_a`.
        warped_b (np.ndarray): The second warped image (canvas size, float32, BGR in [0, 1]).
        mask_b (np.ndarray): Boolean mask indicating valid pixels for `warped_b`.
        n_bands (int, optional): Number of horizontal bands to divide the overlap into. Defaults to 6.
        erode (int, optional): The number of pixels to erode the overlap boundary by. Defaults to 9.
        min_pixels (int, optional): Minimum required number of pixels per band to compute a valid NCC. Defaults to 500.

    Returns:
        list[float]: A list of length `n_bands` containing the NCC values for each band from top to bottom. 
            Returns NaN for bands with insufficient pixels.
    """
    ov = mask_a & mask_b
    if erode > 0:
        ov = cv2.erode(ov.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)).astype(bool)
    rows = np.flatnonzero(ov.any(axis=1))
    if len(rows) == 0:
        return [float("nan")] * n_bands
    hp = lambda g: g - cv2.GaussianBlur(g, (0, 0), 3.0)
    ga, gb = hp(_luminance(warped_a)), hp(_luminance(warped_b))
    edges = np.linspace(rows[0], rows[-1] + 1, n_bands + 1)
    y = np.arange(ov.shape[0])[:, None]
    out = []
    for k in range(n_bands):
        sel = ov & (y >= edges[k]) & (y < edges[k + 1])
        out.append(_ncc(ga[sel], gb[sel]) if sel.sum() >= min_pixels else float("nan"))
    return out


def homography_rms_error(H_est: np.ndarray, H_gt: np.ndarray, image_shape: tuple[int, int], dst_shape: tuple[int, int] | None = None,
                         grid: int = 16) -> float:
    """Computes the root-mean-square (RMS) geometric error between estimated and ground-truth homographies.

    Calculates the RMS distance (in pixels) between points mapped by H_est and H_gt over 
    a regular grid defined on the source image.
    If `dst_shape` is provided, only points that the *ground truth* maps inside the 
    destination image bounds are used. This evaluates the error specifically on the overlap 
    region, which is the only region that matters for stitching.

    Args:
        H_est (np.ndarray): The estimated 3x3 homography matrix.
        H_gt (np.ndarray): The ground-truth 3x3 homography matrix.
        image_shape (tuple[int, int]): The dimensions (height, width) of the source image.
        dst_shape (tuple[int, int] | None, optional): The dimensions (height, width) of the destination image. Defaults to None.
        grid (int, optional): The number of grid points per dimension used for sampling. Defaults to 16.

    Returns:
        float: The computed RMS error in pixels, or NaN if no points fall within the valid evaluation region.
    """
    h, w = image_shape[:2]
    xs, ys = np.meshgrid(np.linspace(0, w - 1, grid), np.linspace(0, h - 1, grid))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    gt = apply_homography(H_gt, pts)
    if dst_shape is not None:
        hd, wd = dst_shape[:2]
        keep = (gt[:, 0] >= 0) & (gt[:, 0] <= wd - 1) & (gt[:, 1] >= 0) & (gt[:, 1] <= hd - 1)
        pts, gt = pts[keep], gt[keep]
    if len(pts) == 0:
        return float("nan")
    d = apply_homography(H_est, pts) - gt
    return float(np.sqrt(np.nanmean((d ** 2).sum(axis=1))))