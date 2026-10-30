"""
Blending of the warped images on the panorama canvas.

Modes:
  overwrite: later images simply cover earlier ones (hard seams - the "naive" baseline).
  average: plain mean in the overlaps (ghosting if alignment is imperfect).
  feather: weights = distance to the image border (soft seam, same transition width for all frequencies).
  multiband: Laplacian-pyramid blending, Burt & Adelson 1983 (each frequency band is blended with a
             mask of matching sharpness). The pyramid operators pyr_down / pyr_up are the ones written
             in Part 1 (src/laplacian_blend.py) -- this is where the two halves of the project meet.

All blending functions take float32 images in [0, 1] of the *canvas* size, (H, W, 3), 
plus boolean masks (H, W) that mark the pixels covered by each image.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from image_blending.laplacian_blend import build_gaussian_pyramid, build_laplacian_pyramid, pyr_up   # Part 1 (src/laplacian_blend.py)

BLEND_MODES = ("overwrite", "average", "feather", "multiband")

#=================================================
# Exposure (gain) compensation
#=================================================

def compute_gains(images: list[np.ndarray], masks: list[np.ndarray], sigma_n: float = 10.0 / 255.0, sigma_g: float = 0.1) -> np.ndarray:
    """Computes multiplicative exposure gains to minimize intensity differences in overlaps.

    Follows the method by Brown & Lowe (2007). Computes one gain `g_i` per image to minimize:
        e = 1/2 sum_i sum_j N_ij [ (g_i I_ij - g_j I_ji)^2 / sigma_n^2 + (1 - g_i)^2 / sigma_g^2 ]
    where `I_ij` is the mean intensity of image i inside its overlap with image j, and `N_ij` 
    is the overlap area. The first term makes overlapping images agree, while the second keeps 
    gains close to 1 (preventing the trivial solution g = 0). 
    Setting de/dg_k = 0 yields a small linear system.

    Args:
        images (list[np.ndarray]): List of warped images (canvas size, float32, [0, 1]).
        masks (list[np.ndarray]): List of boolean masks indicating valid pixels for each image.
        sigma_n (float, optional): Assumed standard deviation of intensity differences. Defaults to 10.0 / 255.0.
        sigma_g (float, optional): Penalty strength for deviating from a gain of 1.0. Defaults to 0.1.

    Returns:
        np.ndarray: A 1D array of calculated gain factors, one for each image.
    """
    n = len(images)
    lum = [im.mean(axis=2) for im in images]
    N, Ibar = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ov = masks[i] & masks[j]
                if ov.any():
                    N[i, j], Ibar[i, j] = ov.sum(), lum[i][ov].mean()
    A, b = np.zeros((n, n)), np.zeros(n)
    for k in range(n):
        for j in range(n):
            if j != k and N[k, j] > 0:
                A[k, k] += N[k, j] * (2.0 * Ibar[k, j] ** 2 / sigma_n ** 2 + 1.0 / sigma_g ** 2)
                A[k, j] -= 2.0 * N[k, j] * Ibar[k, j] * Ibar[j, k] / sigma_n ** 2
                b[k] += N[k, j] / sigma_g ** 2
        if A[k, k] == 0:                        # image without any overlap keeps gain 1
            A[k, k], b[k] = 1.0, 1.0
    return np.linalg.solve(A, b)


#=================================================
# Simple blenders
#=================================================

def blend_overwrite(images: list[np.ndarray], masks: list[np.ndarray], order: list[int] | None = None) -> np.ndarray:
    """Blends images by simply overwriting earlier ones with later ones.

    Produces hard seams. Acts as a naive baseline.

    Args:
        images (list[np.ndarray]): List of images to blend.
        masks (list[np.ndarray]): List of corresponding valid pixel masks.
        order (list[int] | None, optional): Specific indices defining the drawing order. 
            If None, draws in the sequential order of the list. Defaults to None.

    Returns:
        np.ndarray: The blended image array.
    """
    out = np.zeros_like(images[0])
    for k in (order if order is not None else range(len(images))):
        out[masks[k]] = images[k][masks[k]]
    return out


def blend_average(images: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    """Blends images by averaging overlapping regions.

    May produce ghosting artifacts if the alignment is imperfect.

    Args:
        images (list[np.ndarray]): List of images to blend.
        masks (list[np.ndarray]): List of corresponding valid pixel masks.

    Returns:
        np.ndarray: The averaged blended image.
    """
    acc = np.zeros_like(images[0])
    cnt = np.zeros(masks[0].shape, dtype=np.float32)
    for im, m in zip(images, masks):
        acc += im * m[..., None]
        cnt += m
    return acc / np.maximum(cnt, 1.0)[..., None]


def distance_weights(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Computes distance transform weights for image masks.

    Calculates the distance (in pixels) of every covered pixel to the nearest uncovered pixel.
    Values are large in the middle of an image and ~0 at its border. The border of the canvas 
    counts as uncovered, so the distance stays finite even when an image covers the whole canvas 
    (e.g., a close-up that lies completely inside the footprint of another photo).

    Args:
        masks (list[np.ndarray]): List of boolean masks.

    Returns:
        list[np.ndarray]: List of float32 distance maps corresponding to each mask.
    """
    return [cv2.distanceTransform(np.pad(m.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1] for m in masks]


def blend_feather(images: list[np.ndarray], masks: list[np.ndarray], power: float = 1.0) -> np.ndarray:
    """Blends images using a distance-based feathering technique.

    Creates soft seams by weighting pixels based on their distance to the image border. 
    The transition width is the same for all frequencies.

    Args:
        images (list[np.ndarray]): List of images to blend.
        masks (list[np.ndarray]): List of corresponding valid pixel masks.
        power (float, optional): Exponent to apply to the distance weights to adjust 
            the sharpness of the transition. Defaults to 1.0.

    Returns:
        np.ndarray: The feathered blended image.
    """
    w = [d ** power for d in distance_weights(masks)]
    acc = np.zeros_like(images[0])
    for im, wk in zip(images, w):
        acc += im * wk[..., None]
    return acc / np.maximum(sum(w), 1e-6)[..., None]


#=================================================
# Multi-band (Laplacian pyramid) blending
#=================================================

def _fill_invalid(img: np.ndarray, mask: np.ndarray, margin: int) -> np.ndarray:
    """Fills uncovered pixels with the nearest valid color within a margin.

    This prevents the black background from bleeding into the blurred coarse levels 
    and creating dark halos during pyramid blending.

    Args:
        img (np.ndarray): The warped image.
        mask (np.ndarray): The mask indicating valid pixels.
        margin (int): The maximum distance (in pixels) outside the mask to fill.

    Returns:
        np.ndarray: The image with invalid boundary pixels filled with nearest valid colors.
    """
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return img
    y0, y1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, mask.shape[0])
    x0, x1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, mask.shape[1])
    idx = ndimage.distance_transform_edt(~mask[y0:y1, x0:x1], return_distances=False, return_indices=True)
    out = img.copy()
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1][idx[0], idx[1]]
    return out


def blend_multiband(images: list[np.ndarray], masks: list[np.ndarray], num_levels: int = 5) -> np.ndarray:
    """Blends multiple images using Laplacian pyramids (multi-band blending).

    Generalisation of two-image Laplacian blending to K images:
      1. Partition the canvas: each pixel goes to the image with the largest distance-to-border 
         (creates a "hard" mask `M_k` per image, sum_k M_k = 1 on the covered area).
      2. Build a Gaussian pyramid of every mask -> `M_k^(l)` gets blurrier with `l`.
      3. Build a Laplacian pyramid of every image -> band-pass levels `L_k^(l)` (coarsest level = Gaussian residual).
      4. Blend each band with the mask of the same level: `B^(l) = sum_k M_k^(l) L_k^(l) / sum_k M_k^(l)`.
      5. Collapse the pyramid: `I = B^(0) + pyrUp(B^(1) + pyrUp(...))`.

    Low frequencies (exposure, colour) are mixed over a wide transition zone while high-frequency
    detail switches over a narrow one, which hides the seam without blurring the content.

    Args:
        images (list[np.ndarray]): List of warped images to blend.
        masks (list[np.ndarray]): List of valid pixel masks.
        num_levels (int, optional): Number of pyramid levels to compute. Defaults to 5.

    Returns:
        np.ndarray: The multi-band blended image.
    """
    k_img = len(images)
    h, w = masks[0].shape
    dist = np.stack(distance_weights(masks))
    label, covered = dist.argmax(axis=0), dist.max(axis=0) > 0
    hard = [((label == k) & covered).astype(np.float32) for k in range(k_img)]
    del dist

    # every pyramid level must halve exactly -> pad the canvas to a multiple of 2^levels
    mult = 2 ** num_levels
    pad = ((0, (-h) % mult), (0, (-w) % mult))
    margin = 2 ** (num_levels + 2)

    blended = None
    weight_sum = None
    for k in range(k_img):
        img = _fill_invalid(images[k], masks[k], margin)
        img = np.pad(img, pad + ((0, 0),), mode="edge").astype(np.float32)
        gp = build_gaussian_pyramid(img, num_levels) # Part 1: finest -> coarsest
        lp = build_laplacian_pyramid(gp, num_levels) # Part 1: coarsest -> finest
        gm = build_gaussian_pyramid(np.pad(hard[k], pad), num_levels) # mask pyramid, finest -> coarsest
        if blended is None:
            blended = [np.zeros_like(l) for l in lp]
            weight_sum = [np.zeros(l.shape[:2], dtype=np.float32) for l in lp]
        for lev in range(num_levels + 1):
            wk = gm[num_levels - lev].astype(np.float32)
            blended[lev] += lp[lev] * wk[..., None]
            weight_sum[lev] += wk

    for lev in range(num_levels + 1):
        blended[lev] /= np.maximum(weight_sum[lev], 1e-6)[..., None]

    result = blended[0]
    for lev in range(1, num_levels + 1):
        result = pyr_up(result)[: blended[lev].shape[0], : blended[lev].shape[1]] + blended[lev] # Part 1: pyrUp
    return np.clip(result[:h, :w], 0.0, 1.0).astype(np.float32)


#=================================================
# Front end
#=================================================

def blend_images(images: list[np.ndarray], masks: list[np.ndarray], mode: str = "multiband", num_levels: int = 5, 
                 order: list[int] | None = None, gain_compensation: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Top-level function to blend canvas-sized images using the specified mode.

    Args:
        images (list[np.ndarray]): List of warped images (canvas size, float32, [0, 1]).
        masks (list[np.ndarray]): List of boolean masks indicating covered pixels.
        mode (str, optional): The blending mode ("overwrite", "average", "feather", "multiband"). 
            Defaults to "multiband".
        num_levels (int, optional): The number of pyramid levels for multiband blending. 
            Defaults to 5.
        order (list[int] | None, optional): Specific rendering order for the "overwrite" mode. 
            Defaults to None.
        gain_compensation (bool, optional): If True, computes and applies exposure gains 
            before blending. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: A tuple containing:
            - pano (np.ndarray): The final blended panorama image (float32, [0, 1]).
            - coverage (np.ndarray): A boolean mask covering all valid pixels of the panorama.
            - info (dict): A dictionary containing auxiliary information, such as computed 'gains'.
            
    Raises:
        ValueError: If an unknown blending mode is provided.
    """
    if mode not in BLEND_MODES:
        raise ValueError(f"Unknown blend mode '{mode}'. Choose from {BLEND_MODES}.")
    info: dict = {}
    images = list(images)
    if gain_compensation and len(images) > 1:
        gains = compute_gains(images, masks)
        images = [np.clip(im * g, 0.0, 1.0) for im, g in zip(images, gains)]
        info["gains"] = gains

    if mode == "overwrite":
        pano = blend_overwrite(images, masks, order)
    elif mode == "average":
        pano = blend_average(images, masks)
    elif mode == "feather":
        pano = blend_feather(images, masks)
    else:
        pano = blend_multiband(images, masks, num_levels)

    coverage = np.any(np.stack(masks), axis=0)
    pano = np.where(coverage[..., None], pano, 0.0).astype(np.float32)
    return pano, coverage, info