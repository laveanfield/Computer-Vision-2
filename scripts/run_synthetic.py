#!/usr/bin/env python
"""
Controlled robustness study of feature extractors on synthetic views with exact ground truth.

Two views of a real photograph ("the world") are rendered by rotating a virtual camera 
(using `src/panorama/synthetic.py`). One difficulty factor is varied at a time while 
everything else stays at a comfortable default setting:

    overlap:   angle between the two views (decreases overlap)
    roll:      in-plane rotation of the 2nd view (tests rotation invariance)
    scale:     zoom factor of the 2nd view (tests scale invariance)
    noise:     Gaussian sensor noise on both views
    exposure:  gain of the 2nd view (makes it darker and darker)
    blur:      defocus blur of the 2nd view

For every trial and every feature extractor, the *same* matching and RANSAC pipeline is run. 
The estimated homography is then compared against the ground truth on the overlap region.

Usage:
    $ python scripts/run_synthetic.py # Run full study (~10 min) -> data/panorama/result/synthetic/trials.csv
    $ python scripts/run_synthetic.py --trials 2 # Quick test with fewer trials
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from panorama.config import PANO_INPUT_DIR, PANO_RESULT_DIR, StitchConfig
from panorama.features import METHODS
from panorama.metrics import homography_rms_error
from panorama.stitcher import PanoramaStitcher
from panorama.synthetic import SWEEPS, WORLD_WIDTH, render_views, sweep_pair

WORLDS = ["boat/boat1.jpg", "boat/boat2.jpg", "boat/boat4.jpg"]
VIEW_SIZE = (800, 600)


def main() -> None:
    """
    Parses arguments and runs the controlled synthetic robustness study.
    
    Generates simulated view pairs across varying difficulty axes (sweeps), 
    evaluates all feature extractors on these pairs, and records the results 
    to a CSV file.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=6, help="trials per difficulty level (default 6)")
    ap.add_argument("--sweeps", nargs="*", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--out", type=Path, default=PANO_RESULT_DIR / "synthetic" / "trials.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    worlds = []
    for name in WORLDS:
        img = cv2.imread(str(PANO_INPUT_DIR / name))
        worlds.append(cv2.resize(img, (WORLD_WIDTH, int(round(WORLD_WIDTH * img.shape[0] / img.shape[1]))),
                                 interpolation=cv2.INTER_AREA))

    stitchers = {m: PanoramaStitcher(StitchConfig(method=m, max_side=None)) for m in METHODS}
    fields = ["sweep", "level", "trial", "world", "method", "n_kp_a", "n_kp_b", "n_matches", "n_inliers", "inlier_ratio",
              "coverage", "rank_ratio", "valid", "rmse_fit", "gt_err", "t_features", "t_match", "t_ransac"]
    t_start = time.time()
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for sweep in args.sweeps:
            for li, level in enumerate(SWEEPS[sweep][1]):
                for trial in range(args.trials):
                    rng = np.random.default_rng(10_000 * (list(SWEEPS).index(sweep) + 1) + 100 * li + trial)
                    world_id = trial % len(worlds)
                    specs = sweep_pair(sweep, level, rng)
                    views, H_gt, cov = render_views(
                        worlds[world_id], list(specs), size=VIEW_SIZE,
                        seed=int(rng.integers(1 << 30))
                    )
                    if min(cov) < 0.99:
                        print(f"  skip {sweep}={level} trial {trial}: view leaves the world image ({min(cov):.2f})")
                        continue
                    shapes = [v.shape[:2] for v in views]
                    for method, st in stitchers.items():
                        feats = st.compute_features(views)
                        p = st.estimate_pair(feats, shapes, 0, 1)
                        err = (homography_rms_error(p.H, H_gt[(0, 1)], shapes[0], dst_shape=shapes[1])
                               if p.H is not None else float("nan"))
                        writer.writerow(dict(
                            sweep=sweep, level=level, trial=trial, world=world_id, method=method,
                            n_kp_a=len(feats[0]), n_kp_b=len(feats[1]), n_matches=p.n_matches, n_inliers=p.n_inliers,
                            inlier_ratio=round(p.inlier_ratio, 4), coverage=round(p.coverage, 4),
                            rank_ratio=f"{p.rank_ratio:.3e}", valid=int(p.valid),
                            rmse_fit=round(p.rmse, 4) if np.isfinite(p.rmse) else "", gt_err=round(err, 4) if np.isfinite(err) else "",
                            t_features=round(sum(f.seconds for f in feats), 4), t_match=round(p.seconds_match, 4),
                            t_ransac=round(p.seconds_ransac, 4))
                        )
                    fh.flush()
                print(f"[{time.time() - t_start:6.0f}s] {sweep:8s} level {level} done", flush=True)
    print(f"finished in {time.time() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()