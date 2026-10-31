#!/usr/bin/env python
"""
Analyzes the accumulated error (drift) of chained homographies.

The stitcher estimates a homography for the pairs of a spanning tree only and chains 
them to bring every image onto the reference. A pair that overlaps but is NOT an edge 
of the tree gives an independent check: its own homography H_ij (from its own matches) 
and the chain H_j^-1 H_i (through the tree) must agree. Their disagreement is the drift 
the chain has accumulated between i and j; without loop closure or bundle adjustment, 
nothing corrects it.

For every accepted non-tree pair, the script prints the number of tree edges between 
the two images, the share of image i that falls inside image j, and the RMS and maximum 
distance (px) between the points H_ij p and (H_j^-1 H_i) p over a regular grid of 
points p of image i that H_ij maps inside image j. 
Output is sent to stdout and saved to `data/panorama/result/chain_drift.md`.

Example:
    Run all scenes and all extractors (~1.5 min):
        $ python scripts/analyse_chain_drift.py
        
    Run specific scenes and methods:
        $ python scripts/analyse_chain_drift.py --scenes boat newspaper --methods sift
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from panorama.config import PANO_INPUT_DIR, PANO_RESULT_DIR, StitchConfig
from panorama.features import METHODS
from panorama.homography import apply_homography
from panorama.stitcher import PanoramaStitcher, list_images, load_images

ROOT = Path(__file__).resolve().parent.parent


def path_length(tree: list[tuple[int, int, int]], a: int, b: int) -> int:
    """
    Calculates the number of edges on the path between two images in the spanning tree.

    Args:
        tree (list[tuple[int, int, int]]): The spanning tree represented as a list of 
            edges, where each edge is a tuple (node_i, node_j, weight).
        a (int): The index of the first image (node).
        b (int): The index of the second image (node).

    Returns:
        int: The shortest path length (number of edges) between node `a` and node `b`.
    """
    adj: dict[int, list[int]] = {}
    for i, j, _ in tree:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    dist, queue = {a: 0}, deque([a])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist[b]


def drift(
    H_direct: np.ndarray, 
    H_chain: np.ndarray, 
    shape_i: tuple[int, int], 
    shape_j: tuple[int, int],
    grid: int = 40
) -> tuple[float, float, float]:
    """
    Calculates the geometric drift between a direct homography and a chained homography.

    Evaluates the distance between points mapped by `H_direct` and `H_chain` over a 
    regular grid on image `i` that falls within the boundaries of image `j`.

    Args:
        H_direct (np.ndarray): The 3x3 direct homography matrix mapping image `i` to `j`.
        H_chain (np.ndarray): The 3x3 chained homography matrix mapping image `i` to `j`.
        shape_i (tuple[int, int]): The (height, width) dimensions of image `i`.
        shape_j (tuple[int, int]): The (height, width) dimensions of image `j`.
        grid (int, optional): The grid resolution for point sampling. Defaults to 40.

    Returns:
        tuple[float, float, float]: A tuple containing:
            - share (float): The fraction of image `i`'s area that falls inside image `j`.
            - rms (float): The root-mean-square distance in pixels between the mappings.
            - max_dist (float): The maximum distance in pixels between the mappings.
    """
    hi, wi = shape_i[:2]
    hj, wj = shape_j[:2]
    xs, ys = np.meshgrid(np.linspace(0, wi - 1, grid), np.linspace(0, hi - 1, grid))
    p = np.stack([xs.ravel(), ys.ravel()], axis=1)
    q = apply_homography(H_direct, p)
    inside = np.isfinite(q).all(axis=1) & (q[:, 0] >= 0) & (q[:, 0] <= wj - 1) & (q[:, 1] >= 0) & (q[:, 1] <= hj - 1)
    if not inside.any():
        return 0.0, float("nan"), float("nan")
    d = np.linalg.norm(q[inside] - apply_homography(H_chain, p[inside]), axis=1)
    return float(inside.mean()), float(np.sqrt((d ** 2).mean())), float(d.max())


def main() -> None:
    """
    Parses command-line arguments and executes the chain drift analysis.
    
    Reads images, performs panorama stitching to obtain the spanning tree and pairwise 
    homographies, calculates the drift for overlapping non-tree pairs, and writes the 
    results to a Markdown table.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=None, help="default: every folder of the input directory")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--input", type=Path, default=PANO_INPUT_DIR)
    args = ap.parse_args()
    scenes = args.scenes or sorted(p.name for p in args.input.iterdir() if p.is_dir())

    out = [
        "# Drift of the chained homographies against independently estimated pairs", "",
        "| scene | extractor | pair | tree edges between them | share of image i inside j | inliers | RMS drift (px) | max drift (px) |",
        "|:--|:--|:--:|--:|--:|--:|--:|--:|"
    ]
    for scene in scenes:
        images = load_images(list_images(args.input / scene))
        for method in args.methods:
            res = PanoramaStitcher(StitchConfig(method=method, blend="overwrite")).stitch(images)
            if not res.ok:
                print(f"{scene}/{method}: {res.message}")
                continue
            tree_edges = {(min(i, j), max(i, j)) for i, j, _ in res.tree}
            for (i, j), info in sorted(res.pairs.items()):
                if not info.valid or (i, j) in tree_edges or i not in res.homographies or j not in res.homographies:
                    continue
                H_chain = np.linalg.inv(res.homographies[j]) @ res.homographies[i]
                share, rms, mx = drift(info.H, H_chain, res.images[i].shape, res.images[j].shape)
                row = (
                    f"| {scene} | {method} | {i}-{j} | {path_length(res.tree, i, j)} | {100 * share:.0f} % | "
                    f"{info.n_inliers} | {rms:.2f} | {mx:.2f} |"
                )
                out.append(row)
                print(row, flush=True)

    target = PANO_RESULT_DIR / "chain_drift.md"
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"written {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()