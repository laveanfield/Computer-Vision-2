"""
Geometric warping from scratch (NumPy only): bilinear sampling, inverse mapping, canvas layout.

Convention:
    Pixel (row r, column c) is the sample located at the integer coordinate (x = c, y = r), 
    i.e., coordinates refer to pixel *centres*. This is the same convention as OpenCV, 
    so results can be compared with cv2.warpPerspective in the unit tests.

"""
from __future__ import annotations

import numpy as np

from .homography import apply_homography

#=================================================
# Bilinear interpolation
#=================================================

def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples an image at real-valued positions using bilinear interpolation.

    With x0 = floor(x), y0 = floor(y), a = x - x0, b = y - y0, the value is the 
    area-weighted average of the four surrounding pixels:
        f = (1-a)(1-b) I[y0, x0] + a(1-b) I[y0, x0+1] + (1-a)b I[y0+1, x0] + a b I[y0+1, x0+1]

    Args:
        img (np.ndarray): The input image array of shape (H, W) or (H, W, C).
        x (np.ndarray): The x-coordinates (column indices) to sample.
        y (np.ndarray): The y-coordinates (row indices) to sample.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - values (np.ndarray): The sampled pixel values.
            - valid (np.ndarray): A boolean mask indicating where the position falls 
              inside the bounds [0, W-1] x [0, H-1]. Values outside are set to 0.
    """
    img = np.asarray(img)
    H, W = img.shape[:2]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    xs = np.where(valid, x, 0.0)
    ys = np.where(valid, y, 0.0)

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    a = (xs - x0).astype(np.float32)
    b = (ys - y0).astype(np.float32)
    if img.ndim == 3:
        a, b, m = a[..., None], b[..., None], valid[..., None]
    else:
        m = valid

    src = img.astype(np.float32, copy=False)
    out = (
        (1 - a) * (1 - b) * src[y0, x0] + a * (1 - b) * src[y0, x1]
        + (1 - a) * b * src[y1, x0] + a * b * src[y1, x1]
    )
    return np.where(m, out, 0.0).astype(np.float32), valid


#=================================================
# Inverse warping
#=================================================

def warp_perspective(img: np.ndarray, H: np.ndarray, out_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Warps an image into a canvas using a perspective transformation matrix.

    Inverse mapping is used: for every canvas pixel p, the image is sampled at H^-1 p. 
    This guarantees that the output has no holes (forward mapping would leave gaps).

    Args:
        img (np.ndarray): The input image to warp.
        H (np.ndarray): The 3x3 homography matrix mapping image coordinates to canvas coordinates.
        out_shape (tuple[int, int]): The desired output canvas dimensions as (height, width).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - warped (np.ndarray): The warped image as a float32 array of shape (Hc, Wc, C) 
              or (Hc, Wc) depending on the input.
            - mask (np.ndarray): A boolean mask of shape (Hc, Wc) that is True where the 
              canvas pixel is covered by the valid warped image.
    """
    Hc, Wc = out_shape
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    out = np.zeros((Hc, Wc, channels) if img.ndim == 3 else (Hc, Wc), dtype=np.float32)
    mask = np.zeros((Hc, Wc), dtype=bool)

    corners = apply_homography(H, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64))
    if not np.all(np.isfinite(corners)):
        return out, mask
    # only the bounding box of the warped image has to be visited
    x0 = max(int(np.floor(corners[:, 0].min())) - 1, 0)
    x1 = min(int(np.ceil(corners[:, 0].max())) + 2, Wc)
    y0 = max(int(np.floor(corners[:, 1].min())) - 1, 0)
    y1 = min(int(np.ceil(corners[:, 1].max())) + 2, Hc)
    if x1 <= x0 or y1 <= y0:
        return out, mask

    Hinv = np.linalg.inv(H)
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
    q = Hinv[:, 0:1] * xs.ravel() + Hinv[:, 1:2] * ys.ravel() + Hinv[:, 2:3]      # (3, N)
    # points "behind" the camera have the opposite sign of the homogeneous coordinate -> reject them
    sign = np.sign(H[2] @ np.array([(w - 1) / 2.0, (h - 1) / 2.0, 1.0])) or 1.0
    front = (q[2] * sign) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = np.where(front, q[0] / q[2], -1.0).reshape(ys.shape)
        sy = np.where(front, q[1] / q[2], -1.0).reshape(ys.shape)

    vals, valid = bilinear_sample(img, sx, sy)
    out[y0:y1, x0:x1] = vals
    mask[y0:y1, x0:x1] = valid
    return out, mask


#=================================================
# Canvas layout
#=================================================

def compute_canvas(image_shapes: list[tuple[int, int]], homographies: list[np.ndarray], max_side: int | None = 6000) -> tuple[tuple[int, int], np.ndarray]:
    """
    Finds the bounding box and transformation matrix for all warped images.

    Computes the final canvas size needed to encompass all images projected via 
    their respective homographies.

    Args:
        image_shapes (list[tuple[int, int]]): A list of (height, width) tuples for each image.
        homographies (list[np.ndarray]): A list of 3x3 homography matrices mapping each image.
        max_side (int | None, optional): The maximum allowed dimension for the canvas side. 
            If the computed canvas exceeds this, it will be scaled down. Defaults to 6000.

    Returns:
        tuple[tuple[int, int], np.ndarray]: A tuple containing:
            - canvas_shape (tuple[int, int]): The final canvas size as (height, width).
            - T (np.ndarray): The 3x3 translation (and scaling) matrix that adjusts the 
              reference coordinates so everything lies inside [0, W) x [0, H). The final 
              image-to-canvas homography is T @ H_i.

    Raises:
        ValueError: If a homography sends an image corner to infinity, indicating a 
            degenerate projection.
    """
    pts = []
    for (h, w), H in zip(image_shapes, homographies):
        pts.append(apply_homography(H, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)))
    pts = np.concatenate(pts)
    if not np.all(np.isfinite(pts)):
        raise ValueError("A homography sends an image corner to infinity; the projection is degenerate.")

    x_min, y_min = np.floor(pts.min(axis=0))
    x_max, y_max = np.ceil(pts.max(axis=0))
    T = np.array([[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]])
    W, H = int(x_max - x_min + 1), int(y_max - y_min + 1)
    if max_side is not None and max(W, H) > max_side:
        s = max_side / max(W, H)
        T = np.diag([s, s, 1.0]) @ T
        W, H = int(np.ceil(W * s)), int(np.ceil(H * s))
    return (H, W), T