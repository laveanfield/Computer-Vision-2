"""
End-to-end feature-based panorama stitching.

Pipeline:
    load / resize / grey -> features -> pairwise matching + RANSAC homography -> overlap graph
    -> maximum spanning tree, reference image -> chain homographies to the reference
    -> canvas layout -> inverse warping -> (gain compensation) -> blending

The pipeline is independent of the feature extractor: `StitchConfig.method` selects 'sift', 'orb' or 'harris'
and everything else (matching, RANSAC, warping, blending) stays identical, so the extractors can be compared fairly.

"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .blend import blend_images
from .config import IMAGE_EXTENSIONS, StitchConfig
from .features import Features, detect_and_describe
from .homography import dlt_conditioning, footprint_ok, normalize_homography, ransac_homography
from .matching import Matches, match_features
from .metrics import overlap_photometric_error
from .warp import compute_canvas, warp_perspective

MAX_AREA_RATIO = 40.0  # a canvas larger than 40x the total area of the input images means the projection exploded


#=================================================
# Input helpers
#=================================================

def _natural_key(path: Path) -> list:
    """
    Generates a key for natural sorting of strings containing numbers.

    Args:
        path (Path): The file path to parse.

    Returns:
        list: A list of strings and integers for natural sorting comparison.
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_images(folder: str | Path) -> list[Path]:
    """
    Retrieves a list of image files in a directory sorted in natural order.

    Natural order ensures that 'img2' comes before 'img10'.

    Args:
        folder (str | Path): The directory containing the images.

    Returns:
        list[Path]: A list of sorted file paths to the images.
    """
    return sorted(
        (p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()), 
        key=_natural_key
    )


def load_images(paths: list[Path]) -> list[np.ndarray]:
    """
    Loads a sequence of images from the given paths.

    Args:
        paths (list[Path]): A list of file paths to the images.

    Returns:
        list[np.ndarray]: A list of loaded images as BGR NumPy arrays.

    Raises:
        FileNotFoundError: If an image cannot be read from one of the provided paths.
    """
    imgs = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        imgs.append(im)
    return imgs


def preprocess(images: list[np.ndarray], max_side: int | None) -> list[np.ndarray]:
    """
    Preprocesses a list of images by standardizing color and size.

    Converts grayscale images to BGR uint8. Resizes images using area interpolation 
    so that the longer side does not exceed `max_side`.

    Args:
        images (list[np.ndarray]): A list of input images.
        max_side (int | None): The maximum allowed length for the longer side. If None, 
            no resizing is applied.

    Returns:
        list[np.ndarray]: A list of preprocessed BGR uint8 images.
    """
    out = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        if max_side is not None and max(h, w) > max_side:
            s = max_side / max(h, w)
            im = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        out.append(im)
    return out


#=================================================
# Result containers
#=================================================

@dataclass
class PairInfo:
    """
    Outcome of matching and RANSAC for an image pair (i, j), where i < j.

    Attributes:
        i (int): Index of the first image.
        j (int): Index of the second image.
        H (np.ndarray | None): Homography matrix mapping image i onto image j.
        matches (Matches): The feature matches between the two images.
        inliers (np.ndarray): Boolean mask array over `matches` indicating inliers.
        n_inliers (int): Total number of inlier matches.
        iterations (int): Number of RANSAC iterations executed.
        rmse (float): RMS reprojection error of the inliers in pixels.
        coverage (float): Area of the convex hull of the inliers relative to the image area.
        rank_ratio (float): Ratio s_8 / s_1 of the DLT matrix of the inliers. A value near 0 
            indicates a degenerate homography matrix.
        seconds_match (float): Time elapsed during feature matching.
        seconds_ransac (float): Time elapsed during RANSAC homography estimation.
        valid (bool): True if the pair is accepted as overlapping and usable.
    """
    i: int
    j: int
    H: np.ndarray | None
    matches: Matches
    inliers: np.ndarray            
    n_inliers: int
    iterations: int
    rmse: float                    
    coverage: float                
    rank_ratio: float              
    seconds_match: float
    seconds_ransac: float
    valid: bool                    

    @property
    def n_matches(self) -> int:
        """int: The total number of initial feature matches."""
        return len(self.matches)

    @property
    def inlier_ratio(self) -> float:
        """float: The proportion of matches that are inliers."""
        return self.n_inliers / self.n_matches if self.n_matches else 0.0


@dataclass
class PanoramaResult:
    """
    Encapsulates the final results and diagnostics of a panorama stitching operation.

    Attributes:
        ok (bool): Indicates whether the stitching process was successful.
        message (str): Status or error message from the stitching process.
        config (StitchConfig): The configuration used during the process.
        images (list[np.ndarray]): Preprocessed BGR uint8 images that were actually stitched.
        features (list[Features]): Extracted features for each image.
        pairs (dict[tuple[int, int], PairInfo]): Pairwise geometric estimations for overlapping pairs.
        panorama (np.ndarray | None): The final blended panorama image (BGR uint8).
        coverage (np.ndarray | None): Map indicating pixel contribution across the panorama.
        used (list[int]): Indices of the input images included in the final panorama.
        dropped (list[int]): Indices of the input images excluded due to lack of overlap.
        root (int | None): Index of the image used as the central reference frame.
        tree (list[tuple[int, int, int]]): Maximum spanning tree edges represented as (i, j, n_inliers).
        homographies (dict[int, np.ndarray]): Mapping of image indices to their global canvas homographies.
        canvas_shape (tuple[int, int] | None): The computed (height, width) of the final canvas.
        overlap (dict[tuple[int, int], dict]): Photometric error metrics for overlapping image pairs.
        gains (np.ndarray | None): Gain compensation values computed during blending.
        timings (dict): Execution times for different stages of the pipeline.
        warped (list[np.ndarray] | None): Canvas-sized warped images if `keep_intermediate` is True.
        masks (list[np.ndarray] | None): Canvas-sized footprints of the warped images.
        blend_order (list[int] | None): Stacking order used during the blending phase.
    """
    ok: bool
    message: str
    config: StitchConfig
    images: list[np.ndarray]                       
    features: list[Features]
    pairs: dict[tuple[int, int], PairInfo]
    panorama: np.ndarray | None = None             
    coverage: np.ndarray | None = None
    used: list[int] = field(default_factory=list)  
    dropped: list[int] = field(default_factory=list)
    root: int | None = None
    tree: list[tuple[int, int, int]] = field(default_factory=list)       
    homographies: dict[int, np.ndarray] = field(default_factory=dict)    
    canvas_shape: tuple[int, int] | None = None
    overlap: dict[tuple[int, int], dict] = field(default_factory=dict)   
    gains: np.ndarray | None = None
    timings: dict = field(default_factory=dict)
    warped: list[np.ndarray] | None = None         
    masks: list[np.ndarray] | None = None          
    blend_order: list[int] | None = None           


#=================================================
# Graph helpers
#=================================================

def _max_spanning_forest(n: int, edges: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """
    Constructs a maximum spanning forest using Kruskal's algorithm.

    Keeps the heaviest edges (most inliers) that do not create a cycle.

    Args:
        n (int): Number of nodes (images).
        edges (list[tuple[int, int, int]]): List of edges defined as (node_i, node_j, weight).

    Returns:
        list[tuple[int, int, int]]: The subset of edges forming the maximum spanning forest.
    """
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = []
    for i, j, w in sorted(edges, key=lambda e: -e[2]):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((i, j, w))
    return tree


def _components(n: int, tree: list[tuple[int, int, int]]) -> list[list[int]]:
    """
    Finds all connected components within a graph tree.

    Args:
        n (int): Total number of nodes.
        tree (list[tuple[int, int, int]]): The tree edges defined as (node_i, node_j, weight).

    Returns:
        list[list[int]]: A list of components, where each component is a list of node indices.
    """
    adj = {k: [] for k in range(n)}
    for i, j, _ in tree:
        adj[i].append(j)
        adj[j].append(i)
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, q = [], deque([s])
        seen.add(s)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def _hop_distances(adj: dict[int, list[int]], src: int) -> dict[int, int]:
    """
    Computes the shortest path distance (in hops) from a source node to all reachable nodes.

    Args:
        adj (dict[int, list[int]]): Adjacency list representation of the graph.
        src (int): The source node index.

    Returns:
        dict[int, int]: A mapping of node indices to their hop distance from the source.
    """
    dist, q = {src: 0}, deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _hull_fraction(pts: np.ndarray, shape: tuple[int, int]) -> float:
    """
    Calculates the relative area of the convex hull of a set of points.

    Useful for determining if a set of inliers are broadly distributed or collinear.

    Args:
        pts (np.ndarray): 2D array of point coordinates.
        shape (tuple[int, int]): Image dimensions as (height, width).

    Returns:
        float: The convex hull area divided by the total image area. Approaches 0 if points 
            are nearly collinear.
    """
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))) / float(shape[0] * shape[1])


#=================================================
# Stitcher
#=================================================

class PanoramaStitcher:
    """
    Coordinates the end-to-end panoramic stitching pipeline.

    Attributes:
        cfg (StitchConfig): The configuration parameters for stitching.
    """
    def __init__(self, config: StitchConfig | None = None, **overrides):
        """
        Initializes the PanoramaStitcher.

        Args:
            config (StitchConfig | None): Base configuration to use. Defaults to StitchConfig().
            **overrides: Optional key-value pairs to override specific configuration settings.
        """
        self.cfg = replace(config or StitchConfig(), **overrides)

    # stage 1 : features
    def compute_features(self, images: list[np.ndarray]) -> list[Features]:
        """Detects keypoints and extracts descriptors for a list of images.

        Args:
            images (list[np.ndarray]): A list of preprocessed BGR images.

        Returns:
            list[Features]: A list containing the extracted features for each image.
        """
        grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
        return [detect_and_describe(g, self.cfg.method, self.cfg.max_keypoints, self.cfg.fix_coords) for g in grays]

    # stage 2 : pairwise geometry
    def estimate_pair(self, feats: list[Features], shapes: list[tuple[int, int]], i: int, j: int) -> PairInfo:
        """Matches features and estimates a homography between two images.

        Args:
            feats (list[Features]): Extracted features for all images.
            shapes (list[tuple[int, int]]): Shapes (height, width) of all images.
            i (int): Index of the source image.
            j (int): Index of the destination image.

        Returns:
            PairInfo: Diagnostics and homography resulting from the matching and RANSAC process.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        m = match_features(feats[i], feats[j], ratio=cfg.ratio, mutual=cfg.mutual, unique=cfg.unique_matches)
        t1 = time.perf_counter()
        H, inl, info = ransac_homography(feats[i].kpts[m.idx1], feats[j].kpts[m.idx2], thresh=cfg.ransac_thresh,
                                         confidence=cfg.ransac_conf, max_iters=cfg.ransac_max_iters,
                                         seed=cfg.seed + 1000 * i + j)
        t2 = time.perf_counter()
        n_in = int(inl.sum())
        src_in, dst_in = feats[i].kpts[m.idx1[inl]], feats[j].kpts[m.idx2[inl]]
        # Having many inliers is not enough: repeated texture (lines of text, windows, tiles) can produce "inliers"
        # that lie on a line or collapse to a point, and then an entire family of homographies fits them equally well.
        # H is determined iff the DLT matrix of the inliers has rank 8  <=>  s_8 / s_1 is not ~0.
        rank = dlt_conditioning(src_in, dst_in) if n_in >= 4 else 0.0
        cov = min(_hull_fraction(src_in, shapes[i]), _hull_fraction(dst_in, shapes[j])) if n_in >= 3 else 0.0
        plausible = (H is not None and footprint_ok(H, shapes[i], cfg.max_area_ratio)
                     and footprint_ok(np.linalg.inv(H), shapes[j], cfg.max_area_ratio))
        valid = bool(n_in >= cfg.min_inliers and rank >= cfg.min_rank_ratio and plausible)
        return PairInfo(i, j, H, m, inl, n_in, info["iterations"], info["rmse"], cov, rank, t1 - t0, t2 - t1, valid)

    def stitch(self, images: list[np.ndarray], keep_intermediate: bool = False, verbose: bool = False) -> PanoramaResult:
        """
        Executes the full panoramic stitching pipeline on a collection of images.

        Args:
            images (list[np.ndarray]): A list of raw input images.
            keep_intermediate (bool): If True, retains warped images and masks in the result object.
            verbose (bool): If True, prints step-by-step diagnostic information to standard output.

        Returns:
            PanoramaResult: An object encapsulating the final stitched panorama, intermediate states, 
                and performance timings.
        """
        cfg = self.cfg
        t_start = time.perf_counter()
        log = print if verbose else (lambda *a, **k: None)
        timings: dict = {}

        imgs = preprocess(images, cfg.max_side)
        n = len(imgs)
        if n < 2:
            return PanoramaResult(False, "Need at least two images", cfg, imgs, [], {})

        feats = self.compute_features(imgs)
        timings["features"] = sum(f.seconds for f in feats)
        log(f"[{cfg.method}] keypoints per image: {[len(f) for f in feats]}  ({timings['features']:.2f}s)")

        shapes_all = [im.shape[:2] for im in imgs]
        pairs: dict[tuple[int, int], PairInfo] = {}
        for i in range(n):
            for j in range(i + 1, n):
                p = self.estimate_pair(feats, shapes_all, i, j)
                pairs[(i, j)] = p
                log(f"  pair ({i},{j}): matches={p.n_matches:5d} inliers={p.n_inliers:5d} "
                    f"ratio={p.inlier_ratio:.2f} rmse={p.rmse:.2f}px rank={p.rank_ratio:.1e} {'OK' if p.valid else '--'}")
        timings["matching"] = sum(p.seconds_match for p in pairs.values())
        timings["ransac"] = sum(p.seconds_ransac for p in pairs.values())

        # overlap graph -> maximum spanning forest -> biggest connected component
        edges = [(p.i, p.j, p.n_inliers) for p in pairs.values() if p.valid]
        tree_all = _max_spanning_forest(n, edges)
        comps = sorted(_components(n, tree_all), key=lambda c: (-len(c), -sum(w for i, j, w in tree_all if i in c)))
        comp = comps[0]
        if len(comp) < 2:
            return PanoramaResult(
                False, "No pair of images has enough consistent matches "
                f"(>= {cfg.min_inliers} inliers)", cfg, imgs, feats, pairs, timings=timings
            )
        tree = [e for e in tree_all if e[0] in comp]
        adj = {k: [] for k in comp}
        for i, j, _ in tree:
            adj[i].append(j)
            adj[j].append(i)

        # reference image = centre of the tree (smallest maximum hop distance; ties -> most inliers)
        weight_at = {k: sum(w for i, j, w in tree if k in (i, j)) for k in comp}
        root = min(comp, key=lambda k: (max(_hop_distances(adj, k).values()), -weight_at[k]))

        # chain the pairwise homographies to the reference image
        H_root = {root: np.eye(3)}
        queue = deque([root])
        while queue:
            p = queue.popleft()
            for c in adj[p]:
                if c in H_root:
                    continue
                H_cp = pairs[(c, p)].H if (c, p) in pairs else np.linalg.inv(pairs[(p, c)].H)   # maps c -> p
                H_root[c] = normalize_homography(H_root[p] @ H_cp)
                queue.append(c)
        used = sorted(comp)
        dropped = [k for k in range(n) if k not in comp]
        log(f"  reference image: {root}, used: {used}, dropped: {dropped}")

        # canvas layout
        shapes = [imgs[k].shape[:2] for k in used]
        try:
            (hc0, wc0), _ = compute_canvas(shapes, [H_root[k] for k in used], max_side=None)
            if hc0 * wc0 > MAX_AREA_RATIO * sum(h * w for h, w in shapes):
                raise ValueError(f"canvas of {wc0}x{hc0}px is more than {MAX_AREA_RATIO:.0f}x the input area")
            canvas_shape, T = compute_canvas(shapes, [H_root[k] for k in used], max_side=cfg.max_canvas_side)
        except ValueError as e:
            return PanoramaResult(
                False, f"Degenerate projection: {e}", cfg, imgs, feats, pairs, used=used,
                dropped=dropped, root=root, tree=tree, timings=timings
            )
        H_canvas = {k: T @ H_root[k] for k in used}

        # warp
        t0 = time.perf_counter()
        warped, masks = [], []
        for k in used:
            w, m = warp_perspective(imgs[k].astype(np.float32) / 255.0, H_canvas[k], canvas_shape)
            warped.append(w)
            masks.append(m)
        timings["warp"] = time.perf_counter() - t0

        # photometric agreement in the overlaps (before blending)
        overlap = {}
        for a in range(len(used)):
            for b in range(a + 1, len(used)):
                o = overlap_photometric_error(warped[a], masks[a], warped[b], masks[b])
                if o["n_pixels"] >= 500:
                    overlap[(used[a], used[b])] = o

        # blend
        t0 = time.perf_counter()
        hops = _hop_distances(adj, root)
        order = sorted(range(len(used)), key=lambda a: -hops[used[a]]) # farthest first, reference image on top
        pano, coverage, binfo = blend_images(warped, masks, cfg.blend, cfg.num_levels, order, cfg.gain_compensation)
        timings["blend"] = time.perf_counter() - t0

        pano_u8 = np.clip(pano * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if cfg.crop:
            ys, xs = np.nonzero(coverage)
            sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            pano_u8, coverage_c = pano_u8[sl], coverage[sl]
        else:
            coverage_c = coverage
        timings["total"] = time.perf_counter() - t_start

        return PanoramaResult(
            True, "ok", cfg, imgs, feats, pairs, pano_u8, coverage_c, used, dropped, root, tree,
            H_canvas, canvas_shape, overlap, binfo.get("gains"), timings,
            warped if keep_intermediate else None, masks if keep_intermediate else None, order
        )