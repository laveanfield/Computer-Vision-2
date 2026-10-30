"""
Synthetic panorama views with exact ground truth.

A camera that only rotates about its optical centre sees the same scene through the homography:
    x_j ~ H_ij x_i ,        H_ij = K_j R_j^T R_i K_i^-1
(see the derivation in notebooks/demo_panorama.ipynb). We therefore take one large photograph, 
treat it as the image of a pinhole camera ("the world"), and render new views by rotating 
a virtual camera. The true homographies between the views are known, which allows measuring 
the geometric accuracy of a feature extractor in pixels and stressing it in a controlled way 
(rotation angle / overlap, in-plane roll, zoom, noise, exposure).

"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def intrinsics(f: float, width: int, height: int) -> np.ndarray:
    """
    Calculates the camera intrinsic matrix.

    Args:
        f (float): Focal length in pixels.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        np.ndarray: A 3x3 intrinsic camera matrix.
    """
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def rotation_matrix(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """
    Computes the camera-to-world rotation matrix from Euler angles.

    Camera axes are defined as: x right, y down, z forward.

    Args:
        yaw (float, optional): Yaw angle in degrees. Defaults to 0.0.
        pitch (float, optional): Pitch angle in degrees. Defaults to 0.0.
        roll (float, optional): Roll (in-plane rotation) angle in degrees. Defaults to 0.0.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    y, p, r = np.deg2rad([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


@dataclass
class ViewSpec:
    """
    Specification for rendering a synthetic camera view.

    Attributes:
        yaw (float): Yaw angle in degrees. Defaults to 0.0.
        pitch (float): Pitch angle in degrees. Defaults to 0.0.
        roll (float): Roll angle in degrees. Defaults to 0.0.
        f (float | None): Focal length in pixels. If None, it uses the same as the world image. 
            Defaults to None.
        gain (float): Multiplicative exposure change. Defaults to 1.0.
        bias (float): Additive exposure change (grey levels). Defaults to 0.0.
        noise (float): Gaussian noise standard deviation (grey levels). Defaults to 0.0.
        blur (float): Gaussian blur standard deviation (pixels). Defaults to 0.0.
    """
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    f: float | None = None
    gain: float = 1.0
    bias: float = 0.0
    noise: float = 0.0
    blur: float = 0.0


def render_views(
    world: np.ndarray, 
    specs: list[ViewSpec], 
    size: tuple[int, int] = (800, 600),
    f_world: float | None = None, 
    seed: int = 0
) -> tuple[list[np.ndarray], dict[tuple[int, int], np.ndarray], list[float]]:
    """
    Renders one synthetic view per ViewSpec from the provided world image.

    Args:
        world (np.ndarray): The base "world" image as a BGR uint8 array.
        specs (list[ViewSpec]): A list of view specifications to render.
        size (tuple[int, int], optional): The (width, height) of the rendered views. 
            Defaults to (800, 600).
        f_world (float | None, optional): The focal length of the world camera. If None, 
            defaults to the world image width (approx. 53 degrees horizontal FOV). Defaults to None.
        seed (int, optional): Random seed used for noise generation. Defaults to 0.

    Returns:
        tuple[list[np.ndarray], dict[tuple[int, int], np.ndarray], list[float]]: A tuple containing:
            - views (list[np.ndarray]): The rendered views as BGR uint8 images.
            - H_gt (dict[tuple[int, int], np.ndarray]): Ground truth homographies where H_gt[(i, j)] 
              maps view i onto view j.
            - cov (list[float]): Coverage fraction for each view (the fraction of the view that lies 
              inside the valid world image).
    """
    rng = np.random.default_rng(seed)
    hw, ww = world.shape[:2]
    f_world = float(f_world if f_world is not None else ww) # ~53 degrees horizontal field of view
    Ks = intrinsics(f_world, ww, hw)
    wv, hv = size

    views, Ks_v, Rs, cov = [], [], [], []
    for s in specs:
        Kv = intrinsics(s.f if s.f is not None else f_world, wv, hv)
        R = rotation_matrix(s.yaw, s.pitch, s.roll)
        M = Ks @ R @ np.linalg.inv(Kv) # view pixel -> world pixel
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        view = cv2.warpPerspective(world, M, size, flags=flags, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpPerspective(np.full((hw, ww), 255, np.uint8), M, size, flags=flags) > 250
        cov.append(float(valid.mean()))

        v = view.astype(np.float32)
        if s.blur > 0:
            v = cv2.GaussianBlur(v, (0, 0), s.blur)
        v = v * s.gain + s.bias
        if s.noise > 0:
            v = v + rng.normal(0.0, s.noise, v.shape).astype(np.float32)
        views.append(np.clip(v + 0.5, 0, 255).astype(np.uint8))
        Ks_v.append(Kv)
        Rs.append(R)

    H_gt = {}
    for i in range(len(specs)):
        for j in range(len(specs)):
            if i != j:
                H = Ks_v[j] @ Rs[j].T @ Rs[i] @ np.linalg.inv(Ks_v[i])
                H_gt[(i, j)] = H / H[2, 2]
    return views, H_gt, cov


#=================================================
# Controlled difficulty sweeps (used by scripts/run_synthetic.py and the notebook)
#=================================================

WORLD_WIDTH = 1944 # the "world" photographs are resized to this width -> f = width = 53 degrees horizontal FOV

SWEEPS = { # name: (axis label, difficulty levels: easy to hard)
    "overlap": ("Angle between the two views (deg)", [6, 12, 18, 20, 21, 22, 23]),
    "roll": ("In-plane rotation of view 2 (deg)", [0, 45, 90, 135, 180]),
    "scale": ("Zoom factor of view 2", [1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0]),
    "noise": ("Noise sigma (grey levels)", [0, 20, 40, 60, 80, 100]),
    "exposure": ("Gain of view 2", [1.0, 0.6, 0.4, 0.25, 0.15]),
    "blur": ("Blur sigma of view 2 (px)", [0, 1.5, 3, 4.5, 6]),
}


def overlap_fraction(H: np.ndarray, shape_a: tuple[int, int], shape_b: tuple[int, int], grid: int = 40) -> float:
    """
    Calculates the fraction of pixels from image A that map inside image B.

    Uses a regular grid of points to estimate the overlap fraction between two views 
    given their relative homography matrix.

    Args:
        H (np.ndarray): The 3x3 homography matrix mapping points from image A to image B.
        shape_a (tuple[int, int]): The (height, width) of image A.
        shape_b (tuple[int, int]): The (height, width) of image B.
        grid (int, optional): The resolution of the sampling grid on image A. Defaults to 40.

    Returns:
        float: The fraction of sampled points from image A that fall within the bounds of image B.
    """
    ha, wa = shape_a[:2]
    hb, wb = shape_b[:2]
    xs, ys = np.meshgrid(np.linspace(0, wa - 1, grid), np.linspace(0, ha - 1, grid))
    p = np.stack([xs.ravel(), ys.ravel(), np.ones(grid * grid)], axis=0)
    q = H @ p
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = q[0] / q[2], q[1] / q[2]
    ok = (q[2] > 0) & (x >= 0) & (x <= wb - 1) & (y >= 0) & (y <= hb - 1)
    return float(ok.mean())


def nominal_overlap(sweep: str, level: float, size: tuple[int, int] = (800, 600)) -> float:
    """
    Computes the expected overlap fraction for a specific sweep difficulty level.

    Evaluates the overlap without random jitter, representing the ideal nominal conditions.

    Args:
        sweep (str): The name of the parameter sweep (e.g., "overlap", "roll").
        level (float): The difficulty level value for the specified sweep.
        size (tuple[int, int], optional): The (width, height) of the rendered views. 
            Defaults to (800, 600).

    Returns:
        float: The expected overlap fraction (between 0.0 and 1.0) of the two views.
    """
    a, b = sweep_pair(sweep, level, _NoJitter())
    views = [ViewSpec(**{**a.__dict__}), ViewSpec(**{**b.__dict__})]
    K = [intrinsics(v.f if v.f is not None else WORLD_WIDTH, *size) for v in views]
    R = [rotation_matrix(v.yaw, v.pitch, v.roll) for v in views]
    H = K[1] @ R[1].T @ R[0] @ np.linalg.inv(K[0])
    return overlap_fraction(H, (size[1], size[0]), (size[1], size[0]))


class _NoJitter:
    """A mock numpy Generator that always returns the midpoint of the requested range."""
    
    @staticmethod
    def uniform(lo: float, hi: float) -> float:
        """Returns the exact center of the interval [lo, hi].

        Args:
            lo (float): Lower bound.
            hi (float): Upper bound.

        Returns:
            float: The midpoint between `lo` and `hi`.
        """
        return 0.5 * (lo + hi)


def sweep_pair(sweep: str, level: float, rng: np.random.Generator | _NoJitter) -> tuple[ViewSpec, ViewSpec]:
    """
    Generates specifications for a pair of views for a single sweep trial.

    One factor (specified by `sweep`) is set to `level`, while everything else stays at a default 
    state (two views 8 degrees apart). The entire camera rig is randomly jittered using the provided 
    `rng` to ensure that different trials view different parts of the world image.

    Args:
        sweep (str): The name of the difficulty parameter to sweep.
        level (float): The specific level for the parameter.
        rng (np.random.Generator | _NoJitter): The random number generator used for applying jitter.

    Returns:
        tuple[ViewSpec, ViewSpec]: The view specifications for the two synthesized views.

    Raises:
        ValueError: If the provided `sweep` key is not found in the `SWEEPS` configuration.
    """
    j, p = rng.uniform(-3, 3), rng.uniform(-2, 2)
    a, b = ViewSpec(yaw=-4 + j, pitch=p), ViewSpec(yaw=4 + j, pitch=p)
    if sweep == "overlap":
        a, b = ViewSpec(yaw=-level / 2 + j, pitch=p), ViewSpec(yaw=level / 2 + j, pitch=p)
    elif sweep == "roll":
        b.roll = level
    elif sweep == "scale":
        a, b = ViewSpec(yaw=-1 + j, pitch=p), ViewSpec(yaw=1 + j, pitch=p, f=WORLD_WIDTH * level)
    elif sweep == "noise":
        a.noise = b.noise = level
    elif sweep == "exposure":
        b.gain = level
    elif sweep == "blur":
        b.blur = level
    elif sweep not in SWEEPS:
        raise ValueError(f"unknown sweep '{sweep}'")
    return a, b