"""
Harris corners + Multi-Scale Oriented Patches (MOPS), written from scratch.

This is the "classic hand-made" feature pipeline used as a third, fully transparent baseline next to
OpenCV's SIFT and ORB (Brown, Szeliski & Winder, "Multi-image matching using multi-scale oriented
patches", CVPR 2005):
  1. Gaussian pyramid (factor 2, sigma 1) -> a coarse form of scale invariance.
  2. Harris structure tensor at every level
         M = G_{sigma_i} * [ Ix^2  IxIy ; IxIy  Iy^2 ]      corner strength  f_HM = det(M) / trace(M)
     (harmonic mean of the eigenvalues: large only if the gradient varies in *both* directions).
  3. 3x3 non-maximum suppression + sub-pixel refinement (parabola fit).
  4. Adaptive non-maximal suppression (ANMS): keep the points with the largest suppression radius so that the
     features are spread evenly over the image instead of clustering on a few textured spots.
  5. Orientation from the gradient of a heavily blurred image (sigma_o = 4.5) -> rotation invariance.
  6. Descriptor: an 8x8 patch sampled with spacing 5 px from a 40x40 oriented window of the (blurred) level
     image, normalised to zero mean / unit variance (gain & bias invariance).  64 floats, matched with L2.
"""
from __future__ import annotations

import cv2
import numpy as np

from .warp import bilinear_sample

PATCH = 8 # descriptor is PATCH x PATCH samples
SPACING = 5.0 # distance (in level pixels) between two samples
WINDOW = PATCH * SPACING                       # 40 px
MARGIN = int(np.ceil(0.5 * (PATCH - 1) * SPACING * np.sqrt(2))) + 2     # window half-diagonal + interpolation

#=================================================
# Corner strength, peaks, ANMS
#=================================================

def harris_response(img: np.ndarray, sigma_d: float = 1.0, sigma_i: float = 1.5, mode: str = "harmonic") -> np.ndarray:
    """
    Calculates the corner strength map of an image using the Harris structure tensor.

    Args:
        img (np.ndarray): The input image as a 2D float array.
        sigma_d (float, optional): Gaussian blur standard deviation for the image gradient. Defaults to 1.0.
        sigma_i (float, optional): Gaussian blur standard deviation for the structure tensor components. Defaults to 1.5.
        mode (str, optional): The scoring mode to use. 
            'harmonic' returns det/trace (MOPS), while 'harris' returns det - 0.04 * trace^2. 
            Defaults to "harmonic".

    Returns:
        np.ndarray: The computed corner strength map.
    """
    g = cv2.GaussianBlur(img, (0, 0), sigma_d)
    gy, gx = np.gradient(g)
    sxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_i)
    syy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_i)
    sxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_i)
    det, tr = sxx * syy - sxy * sxy, sxx + syy
    return det / (tr + 1e-12) if mode == "harmonic" else det - 0.04 * tr * tr


def _peaks(f: np.ndarray, thresh_rel: float, max_candidates: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds local maxima in a corner strength map and refines them to sub-pixel accuracy.

    Applies a 3x3 non-maximum suppression and fits a parabola to refine the peak locations.

    Args:
        f (np.ndarray): The 2D corner strength map.
        thresh_rel (float): The relative threshold below which peaks are ignored.
        max_candidates (int): The maximum number of peaks to return.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - xs (np.ndarray): Refined x-coordinates of the peaks.
            - ys (np.ndarray): Refined y-coordinates of the peaks.
            - fc (np.ndarray): The corner strength values at the peaks.
    """
    local_max = f >= cv2.dilate(f, np.ones((3, 3), np.uint8))
    thr = max(thresh_rel * float(f.max()), 1e-9)
    ok = local_max & (f > thr)
    ok[:MARGIN], ok[-MARGIN:], ok[:, :MARGIN], ok[:, -MARGIN:] = False, False, False, False
    ys, xs = np.nonzero(ok)
    if len(ys) > max_candidates:
        keep = np.argpartition(-f[ys, xs], max_candidates - 1)[:max_candidates]
        ys, xs = ys[keep], xs[keep]

    fc = f[ys, xs]
    dx_den = f[ys, xs - 1] - 2 * fc + f[ys, xs + 1]
    dy_den = f[ys - 1, xs] - 2 * fc + f[ys + 1, xs]
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(dx_den < 0, 0.5 * (f[ys, xs - 1] - f[ys, xs + 1]) / dx_den, 0.0)
        dy = np.where(dy_den < 0, 0.5 * (f[ys - 1, xs] - f[ys + 1, xs]) / dy_den, 0.0)
    return xs + np.clip(dx, -0.5, 0.5), ys + np.clip(dy, -0.5, 0.5), fc


def anms(xy: np.ndarray, strength: np.ndarray, n_keep: int, c_robust: float = 0.9, chunk: int = 256) -> np.ndarray:
    """Performs Adaptive Non-Maximal Suppression (ANMS) on a set of keypoints.

    Point i is suppressed by every point j that is clearly stronger (c_robust * f_j > f_i).
    Its suppression radius r_i is the distance to the nearest such point. The n_keep
    points with the largest radius are returned to ensure features are spread evenly 
    over the image.

    Args:
        xy (np.ndarray): The coordinates of the keypoints as an (N, 2) array.
        strength (np.ndarray): The corner strength of each keypoint as an (N,) array.
        n_keep (int): The number of top keypoints to keep based on their suppression radius.
        c_robust (float, optional): The robustness multiplier used to determine if a point 
            is significantly stronger. Defaults to 0.9.
        chunk (int, optional): The batch size used for distance calculations to save memory. 
            Defaults to 256.

    Returns:
        np.ndarray: Indices of the selected keypoints that maximize the suppression radius.
    """
    m = len(xy)
    if m <= n_keep:
        return np.arange(m)
    order = np.argsort(-strength, kind="stable")
    p, f = xy[order].astype(np.float32), strength[order].astype(np.float64)
    # points are sorted by decreasing strength, so the suppressors of i are a prefix [0, J_i)
    prefix = np.searchsorted(-f, -f / c_robust, side="left")        # number of j with f_j > f_i / c
    radius = np.full(m, np.inf, dtype=np.float32)
    cols = np.arange(m)[None, :]
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        d2 = ((p[s:e, None, :] - p[None, :, :]) ** 2).sum(-1)
        d2[cols >= prefix[s:e, None]] = np.inf
        radius[s:e] = d2.min(axis=1)
    top = np.argsort(-radius, kind="stable")[:n_keep]
    return order[top]


#=================================================
# Orientation + descriptor
#=================================================

def _describe_level(level: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Extracts oriented and normalized 8x8 patches (descriptors) for a set of keypoints.

    Computes the dominant orientation from a heavily blurred gradient, rotates the 
    sampling grid accordingly, and extracts a normalized patch for gain and bias invariance.

    Args:
        level (np.ndarray): The image level array from the Gaussian pyramid.
        xy (np.ndarray): The (x, y) coordinates of the keypoints in the given level's coordinate space.

    Returns:
        np.ndarray: The extracted descriptors as an (N, 64) float32 array.
    """
    oy, ox = np.gradient(cv2.GaussianBlur(level, (0, 0), 4.5))
    gx, _ = bilinear_sample(ox, xy[:, 0], xy[:, 1])
    gy, _ = bilinear_sample(oy, xy[:, 0], xy[:, 1])
    theta = np.arctan2(gy, gx)
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]

    off = (np.arange(PATCH) - 0.5 * (PATCH - 1)) * SPACING
    gxo, gyo = np.meshgrid(off, off)
    gxo, gyo = gxo.ravel()[None, :], gyo.ravel()[None, :]
    px = xy[:, 0:1] + c * gxo - s * gyo                        # rotate the sampling grid by theta
    py = xy[:, 1:2] + s * gxo + c * gyo

    patch, _ = bilinear_sample(cv2.GaussianBlur(level, (0, 0), 2.0), px, py)          # (n, 64)
    patch = patch - patch.mean(axis=1, keepdims=True)                                 # bias invariance
    patch = patch / (patch.std(axis=1, keepdims=True) + 1e-6)                           # gain invariance
    return patch.astype(np.float32)


def detect_harris_mops(gray: np.ndarray, max_keypoints: int = 4000, levels: int = 4, candidates_per_level: int = 3000,
                       thresh_rel: float = 0.01, c_robust: float = 0.9) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detects Harris corners and computes MOPS descriptors over a Gaussian pyramid.

    Builds an image pyramid, detects corner peaks on each level, applies ANMS to filter 
    and spread out the features, and finally extracts rotation- and scale-invariant 
    descriptors for the remaining keypoints.

    Args:
        gray (np.ndarray): The input grayscale image.
        max_keypoints (int, optional): The maximum total number of keypoints to return. Defaults to 4000.
        levels (int, optional): The number of pyramid levels to build and search. Defaults to 4.
        candidates_per_level (int, optional): The max number of peak candidates to consider per level 
            before ANMS. Defaults to 3000.
        thresh_rel (float, optional): The relative threshold for peak detection on the corner 
            strength map. Defaults to 0.01.
        c_robust (float, optional): The robustness multiplier used in the ANMS step. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - keypoints (np.ndarray): (N, 2) array of float32 [x, y] coordinates in full-resolution pixels.
            - descriptors (np.ndarray): (N, 64) array of float32 MOPS descriptors.
            - window_size (np.ndarray): (N,) array of float32 representing the effective window size.
            - strength (np.ndarray): (N,) array of float32 representing the corner strength.
    """
    img = gray.astype(np.float32)
    if gray.dtype == np.uint8:
        img /= 255.0

    pyramid = [img]
    for _ in range(levels - 1):
        pyramid.append(cv2.GaussianBlur(pyramid[-1], (0, 0), 1.0)[::2, ::2].copy())

    xs, ys, fs, lv = [], [], [], []
    for l, im in enumerate(pyramid):
        if min(im.shape) < 2 * MARGIN + 3:
            break
        x, y, f = _peaks(harris_response(im), thresh_rel, candidates_per_level)
        xs.append(x); ys.append(y); fs.append(f); lv.append(np.full(len(x), l))
    if not xs:
        return (np.zeros((0, 2), np.float32), np.zeros((0, PATCH * PATCH), np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.float32))
    x, y, f, lvl = (np.concatenate(a) for a in (xs, ys, fs, lv))

    full_xy = np.stack([x, y], axis=1) * (2.0 ** lvl)[:, None]          # level-l pixel -> full-resolution pixel
    keep = anms(full_xy, f, max_keypoints, c_robust)
    x, y, f, lvl, full_xy = x[keep], y[keep], f[keep], lvl[keep], full_xy[keep]

    desc = np.zeros((len(keep), PATCH * PATCH), dtype=np.float32)
    for l in np.unique(lvl):
        sel = np.flatnonzero(lvl == l)
        desc[sel] = _describe_level(pyramid[l], np.stack([x[sel], y[sel]], axis=1))
    return (full_xy.astype(np.float32), desc, (WINDOW * 2.0 ** lvl).astype(np.float32), f.astype(np.float32))