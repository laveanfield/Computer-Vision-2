"""Feature matching from scratch (NumPy only).

Implements brute-force 2-nearest-neighbours search, Lowe's ratio test, and an optional 
mutual (cross-check) matching strategy.

Distances:
    - Float descriptors (SIFT, Harris-MOPS): Euclidean distance.
      ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b  -> Computed via a single matrix product for all pairs.
    - Binary descriptors (ORB): Hamming distance.
      For bit vectors a, b in {0,1}^n:
      Hamming(a, b) = sum_i |a_i - b_i| = sum_i (a_i - b_i)^2 = ||a - b||^2.
      This allows using the exact same matrix product trick after unpacking the bits into floats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import Features

#=================================================
# Distances and k-NN
#=================================================

def _prepare(desc: np.ndarray, binary: bool) -> np.ndarray:
    """Prepares descriptors for distance calculation.

    If descriptors are binary, unpacks the bits into an array of float32 zeros and ones.
    Otherwise, simply ensures they are cast to float32.

    Args:
        desc (np.ndarray): The input descriptor array.
        binary (bool): True if descriptors are binary (e.g., ORB), False if float (e.g., SIFT).

    Returns:
        np.ndarray: The prepared float32 descriptor array.
    """
    if binary:
        return np.unpackbits(np.asarray(desc, dtype=np.uint8), axis=1).astype(np.float32)
    return np.asarray(desc, dtype=np.float32)


def knn2(desc1: np.ndarray, desc2: np.ndarray, binary: bool, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Finds the 2 nearest neighbors in `desc2` for every descriptor in `desc1`.

    Rows are processed in chunks to limit the memory footprint of the intermediate 
    distance matrix.

    Args:
        desc1 (np.ndarray): Query descriptor array of shape (N1, D).
        desc2 (np.ndarray): Train descriptor array of shape (N2, D).
        binary (bool): Indicates if the descriptors are binary (uses Hamming distance).
        chunk (int, optional): Number of query descriptors to process at a time. Defaults to 1024.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - indices (np.ndarray): Array of shape (N1, 2) with the indices of the 2 nearest neighbors.
            - distances (np.ndarray): Array of shape (N1, 2) with the corresponding distances.
    """
    a, b = _prepare(desc1, binary), _prepare(desc2, binary)
    n1, n2 = len(a), len(b)
    if n2 < 2:
        return np.zeros((n1, 2), dtype=np.int64), np.full((n1, 2), np.inf, dtype=np.float32)
    b2 = (b * b).sum(axis=1)
    idx = np.empty((n1, 2), dtype=np.int64)
    dist = np.empty((n1, 2), dtype=np.float32)

    for s in range(0, n1, chunk):
        blk = a[s:s + chunk]
        d2 = (blk * blk).sum(axis=1)[:, None] + b2[None, :] - 2.0 * (blk @ b.T)
        np.maximum(d2, 0.0, out=d2)
        part = np.argpartition(d2, 1, axis=1)[:, :2]                       # two smallest (unordered)
        vals = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(vals, axis=1)
        part, vals = np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)
        idx[s:s + chunk] = part
        # Hamming distance = squared L2 of the bit vectors; Euclidean distance = sqrt of the squared L2
        dist[s:s + chunk] = np.rint(vals) if binary else np.sqrt(vals)
    return idx, dist


#=================================================
# Matching
#=================================================

@dataclass
class Matches:
    """Container for putative feature correspondences between two images.

    All index and distance arrays are strictly aligned (same length M).

    Attributes:
        idx1 (np.ndarray): (M,) array of indices into the first feature set.
        idx2 (np.ndarray): (M,) array of indices into the second feature set.
        dist (np.ndarray): (M,) array containing the distance to the nearest neighbor.
        ratio (np.ndarray): (M,) array containing the ratio d_nearest / d_second, 
            which is the value thresholded by Lowe's test.
        n_queries (int): The total number of features in the first (query) image.
    """
    idx1: np.ndarray        
    idx2: np.ndarray        
    dist: np.ndarray        
    ratio: np.ndarray       
    n_queries: int          

    def __len__(self) -> int:
        """Returns the number of matching pairs."""
        return len(self.idx1)


def match_features(f1: Features, f2: Features, ratio: float = 0.75, mutual: bool = False,
                   unique: bool = True) -> Matches:
    """Matches features from `f1` to `f2` using nearest-neighbor search and filters.

    Applies Lowe's ratio test: d1 / d2 < ratio (where d1, d2 are distances to the best 
    and second-best candidates). Ambiguous features where the second neighbor is almost 
    as close as the first have a ratio near 1 and are discarded.

    Args:
        f1 (Features): Query features extracted from the first image.
        f2 (Features): Target features extracted from the second image.
        ratio (float, optional): Threshold for Lowe's ratio test. Defaults to 0.75.
        mutual (bool, optional): If True, strictly keeps matches that are also nearest 
            neighbors when searching in reverse (f2 -> f1). Defaults to False.
        unique (bool, optional): If True, enforces one-to-one assignment. When several 
            features in f1 survive the ratio test pointing to the *same* partner in f2 
            (common in repeated textures), only the most distinctive one (smallest ratio) 
            is kept. Defaults to True.

    Returns:
        Matches: A Matches object containing the filtered feature correspondences.

    Raises:
        ValueError: If attempting to match binary descriptors with float descriptors.
    """
    if f1.binary != f2.binary:
        raise ValueError("Cannot match binary and float descriptors")
    if len(f1) == 0 or len(f2) < 2:
        e = np.zeros(0, dtype=np.int64)
        return Matches(e, e, np.zeros(0, np.float32), np.zeros(0, np.float32), len(f1))

    idx, dist = knn2(f1.desc, f2.desc, f1.binary)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dist[:, 0] / dist[:, 1]
    keep = np.isfinite(r) & (r < ratio)

    if mutual:
        back, _ = knn2(f2.desc, f1.desc, f1.binary)
        keep &= back[idx[:, 0], 0] == np.arange(len(f1))

    q = np.flatnonzero(keep)
    if unique and len(q) > 1:
        order = np.argsort(r[q], kind="stable")                     # most distinctive first
        _, first = np.unique(idx[q[order], 0], return_index=True)   # first occurrence = smallest ratio per partner
        q = np.sort(q[order[first]])                                # back to query order
    return Matches(q, idx[q, 0], dist[q, 0], r[q].astype(np.float32), len(f1))