#!/usr/bin/env python
"""
Stitches images for scenes using multiple feature extractors and records the results.

This script processes scenes found in `data/panorama/input/` (or a custom directory). 
It runs the exact same stitching pipeline (preprocess -> features -> matching -> 
RANSAC -> chain -> warp -> multi-band blend) for each specified feature extractor by 
only changing the `StitchConfig.method` parameter.

Outputs (saved to `data/panorama/result/`):
    <scene>/<scene>_<method>.jpg        The generated panoramas.
    scenes.csv                          Summary row per (scene, extractor) detailing keypoints, 
                                        matches, inliers, RMSE, photometric agreement, and stage run times.
    pairs.csv                           Detailed row per (scene, extractor, image pair) indicating acceptance.
    figures/keypoints_<scene>.jpg       Visual comparison of the keypoints detected by each extractor.
    figures/matches_<scene>.jpg         Visual comparison of matching performance on a single image pair.
    figures/panorama_<scene>.jpg        Contact sheet of the resulting panoramas for comparison.
    figures/blend_<scene>.jpg           Visual comparison of different blending modes along a seam.

Usage:
    $ python scripts/run_scenes.py                                 # Run all scenes with all extractors (~2 min)
    $ python scripts/run_scenes.py --scenes boat mountain --methods sift orb
    $ python scripts/run_scenes.py --input path/to/my_scenes       # Run on custom photos (one sub-folder per scene)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from panorama import viz
from panorama.config import PANO_INPUT_DIR, PANO_RESULT_DIR, StitchConfig
from panorama.features import METHODS
from panorama.stitcher import PanoramaStitcher, list_images, load_images

NAN = float("nan")

SCENE_FIELDS = [
    "scene", "method", "n_images", "ok", "n_used", "n_dropped", "kp_mean", "tree_pairs",
    "matches_tree", "inliers_tree", "inlier_ratio", "rmse_tree",
    "ncc_hp_tree", "ncc_hp_other", "mae_tree", "mae_other", "ncc_tree",
    "t_features", "t_matching", "t_ransac", "t_warp", "t_blend", "t_total",
    "pano_w", "pano_h", "message"
]
PAIR_FIELDS = [
    "scene", "method", "i", "j", "n_matches", "n_inliers", "inlier_ratio", "rmse", "rank_ratio", "valid",
    "in_tree", "ncc_hp", "mae"
]


def _mean(values: list[float]) -> float:
    """
    Calculates the mean of a list of finite values, returning NaN if empty.

    Args:
        values (list[float]): A list of numerical values.

    Returns:
        float: The computed mean or NaN.
    """
    v = [x for x in values if np.isfinite(x)]
    return float(np.mean(v)) if v else NAN


def scene_row(scene: str, method: str, res: PanoramaStitcher.ResultType) -> dict:
    """
    Extracts summary metrics for an entire scene into a dictionary row.

    Args:
        scene (str): The name of the scene being processed.
        method (str): The feature extraction method used.
        res: The result object from the PanoramaStitcher.

    Returns:
        dict: A dictionary containing metrics conforming to SCENE_FIELDS.
    """
    tree = {(i, j) for i, j, _ in res.tree}
    ps = [res.pairs[e] for e in sorted(tree)]
    ov_tree = [res.overlap[e] for e in sorted(tree) if e in res.overlap]
    ov_other = [o for e, o in res.overlap.items() if e not in tree]
    n_m, n_i = sum(p.n_matches for p in ps), sum(p.n_inliers for p in ps)
    t = res.timings
    return dict(
        scene=scene, method=method, n_images=len(res.images), ok=int(res.ok), n_used=len(res.used), n_dropped=len(res.dropped),
        kp_mean=round(float(np.mean([len(f) for f in res.features])), 1) if res.features else NAN,
        tree_pairs=len(ps), matches_tree=n_m, inliers_tree=n_i, inlier_ratio=round(n_i / n_m, 4) if n_m else NAN,
        rmse_tree=round(_mean([p.rmse for p in ps]), 4),
        ncc_hp_tree=round(_mean([o["ncc_hp"] for o in ov_tree]), 4), ncc_hp_other=round(_mean([o["ncc_hp"] for o in ov_other]), 4),
        mae_tree=round(_mean([o["mae"] for o in ov_tree]), 3), mae_other=round(_mean([o["mae"] for o in ov_other]), 3),
        ncc_tree=round(_mean([o["ncc"] for o in ov_tree]), 4),
        t_features=round(t.get("features", NAN), 3), t_matching=round(t.get("matching", NAN), 3),
        t_ransac=round(t.get("ransac", NAN), 3), t_warp=round(t.get("warp", NAN), 3), t_blend=round(t.get("blend", NAN), 3),
        t_total=round(t.get("total", NAN), 3),
        pano_w=res.panorama.shape[1] if res.ok else 0, pano_h=res.panorama.shape[0] if res.ok else 0,
        message=res.message
    )


def pair_rows(scene: str, method: str, res: PanoramaStitcher.ResultType) -> list[dict]:
    """
    Extracts detailed metrics for every image pair evaluated into dictionary rows.

    Args:
        scene (str): The name of the scene being processed.
        method (str): The feature extraction method used.
        res: The result object from the PanoramaStitcher.

    Returns:
        list[dict]: A list of dictionaries, each containing metrics for an image pair 
            conforming to PAIR_FIELDS.
    """
    tree = {(i, j) for i, j, _ in res.tree}
    rows = []
    for (i, j), p in sorted(res.pairs.items()):
        o = res.overlap.get((i, j), {})
        rows.append(
            dict(
                scene=scene, method=method, i=i, j=j, n_matches=p.n_matches, n_inliers=p.n_inliers,
                inlier_ratio=round(p.inlier_ratio, 4), rmse=round(p.rmse, 4) if np.isfinite(p.rmse) else "",
                rank_ratio=f"{p.rank_ratio:.3e}", valid=int(p.valid), in_tree=int((i, j) in tree),
                ncc_hp=round(o["ncc_hp"], 4) if o else "", mae=round(o["mae"], 3) if o else ""
            )
        )
    return rows


def main() -> None:
    """Parses command line arguments, executes the stitching pipeline, and generates outputs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=PANO_INPUT_DIR, help="folder with one sub-folder per scene")
    ap.add_argument("--out", type=Path, default=PANO_RESULT_DIR, help="result folder")
    ap.add_argument("--scenes", nargs="*", default=None, help="scene names (default: all sub-folders)")
    ap.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--max-side", type=int, default=1000, help="resize the images so that the longer side is at most this")
    ap.add_argument("--blend", default="multiband", choices=["overwrite", "average", "feather", "multiband"])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    scenes = args.scenes or sorted(p.name for p in args.input.iterdir() if p.is_dir())
    fig_dir = args.out / "figures"
    args.out.mkdir(parents=True, exist_ok=True)
    all_scene_rows, all_pair_rows = [], []
    t_start = time.time()

    for scene in scenes:
        paths = list_images(args.input / scene)
        imgs = load_images(paths)
        print(f"\n=== {scene}: {len(imgs)} images ({', '.join(p.name for p in paths)})")
        results = {}
        for method in args.methods:
            cfg = StitchConfig(method=method, max_side=args.max_side, blend=args.blend)
            res = PanoramaStitcher(cfg).stitch(imgs, keep_intermediate=(method == "sift"))
            results[method] = res
            all_scene_rows.append(scene_row(scene, method, res))
            all_pair_rows += pair_rows(scene, method, res)
            r = all_scene_rows[-1]
            print(f"  {method:7s} ok={res.ok} used={res.used} kp={r['kp_mean']:.0f} inliers={r['inliers_tree']} "
                  f"ratio={r['inlier_ratio']:.2f} rmse={r['rmse_tree']:.2f}px ncc_hp={r['ncc_hp_tree']:.3f} "
                  f"t={r['t_total']:.1f}s  {'' if res.ok else res.message}")
            if res.ok:
                out = args.out / scene / f"{scene}_{method}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), res.panorama, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if args.no_figures or not all(results[m].ok for m in results):
            continue
        viz.apply_style()
        ref = results[args.methods[0]]
        # 1. keypoints of the first image
        viz.save_figure(
            viz.keypoint_figure(ref.images[0], {m: results[m].features[0] for m in results}),
            fig_dir / f"keypoints_{scene}.jpg", dpi=120
        )
        # 2. the same pair, matched by every extractor (the hardest pair of the SIFT spanning tree)
        base = results.get("sift", ref)
        i, j, _ = min(base.tree, key=lambda e: e[2])
        per = {}
        for m, res in results.items():
            p, f = res.pairs[(i, j)], res.features
            per[m] = dict(
                pts_a=f[i].kpts[p.matches.idx1], pts_b=f[j].kpts[p.matches.idx2], inliers=p.inliers,
                n_matches=p.n_matches, n_inliers=p.n_inliers, rmse=p.rmse if np.isfinite(p.rmse) else 0.0
            )
        viz.save_figure(
            viz.matches_figure(ref.images[i], ref.images[j], per, title=f"{scene}: images {i} and {j}"),
            fig_dir / f"matches_{scene}.jpg", dpi=110
        )
        # 3. the panoramas
        rows = {m: next(x for x in all_scene_rows if x["scene"] == scene and x["method"] == m) for m in results}
        caps = {
            m: f"{viz.SERIES[m]['label']}\n{r['inliers_tree']} inliers ({100 * r['inlier_ratio']:.0f} %), "
            f"RMSE {r['rmse_tree']:.2f} px, NCC {r['ncc_hp_tree']:.2f}" for m, r in rows.items()
        }
        viz.save_figure(
            viz.panorama_figure({m: results[m].panorama for m in results}, caps),
            fig_dir / f"panorama_{scene}.jpg", dpi=110
        )
        # 4. blending modes at a seam (SIFT result)
        if base.warped is not None and len(base.used) >= 2:
            viz.save_figure(viz.blend_comparison_figure(base), fig_dir / f"blend_{scene}.jpg", dpi=120)

    with open(args.out / "scenes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCENE_FIELDS)
        w.writeheader()
        w.writerows(all_scene_rows)
    with open(args.out / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        w.writeheader()
        w.writerows(all_pair_rows)
    print(f"\nfinished in {time.time() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()