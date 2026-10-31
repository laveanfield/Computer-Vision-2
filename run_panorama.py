#!/usr/bin/env python
"""
Command-line front end of Part 2: stitch a panorama from a folder of overlapping photos.

    python run_panorama.py --scene boat                              # a sample scene of data/panorama/input/
    python run_panorama.py --scene boat --method orb --blend feather
    python run_panorama.py --images a.jpg b.jpg c.jpg --method sift --gain --out my_panorama.jpg
    python run_panorama.py --folder path/to/my/photos --compare      # all three extractors, one table

The images may be given in any order: the overlap graph is estimated from the matches, the central image becomes
the reference and images that do not overlap with the rest are dropped (and reported).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from panorama.config import PANO_INPUT_DIR, PANO_RESULT_DIR, StitchConfig
from panorama.features import METHODS
from panorama.stitcher import PanoramaStitcher, list_images, load_images

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help=f"name of a sub-folder of {PANO_INPUT_DIR.relative_to(ROOT)}")
    src.add_argument("--folder", type=Path, help="any folder of images")
    src.add_argument("--images", nargs="+", type=Path, help="explicit list of image files")
    ap.add_argument("--method", default="sift", choices=METHODS, help="feature extractor (default: sift)")
    ap.add_argument("--compare", action="store_true", help="run all extractors and print a comparison table")
    ap.add_argument("--blend", default="multiband", choices=["overwrite", "average", "feather", "multiband"])
    ap.add_argument("--levels", type=int, default=5, help="pyramid levels of the multi-band blender")
    ap.add_argument("--gain", action="store_true", help="exposure (gain) compensation before blending")
    ap.add_argument("--max-side", type=int, default=1000, help="resize so that the longer side is at most this (0 = keep)")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio-test threshold")
    ap.add_argument("--thresh", type=float, default=3.0, help="RANSAC inlier threshold in pixels")
    ap.add_argument("--no-crop", action="store_true", help="keep the black border of the canvas")
    ap.add_argument("--out", type=Path, default=None, help="output file (default: data/panorama/result/<scene>/...)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.scene:
        paths, name = list_images(PANO_INPUT_DIR / args.scene), args.scene
    elif args.folder:
        paths, name = list_images(args.folder), args.folder.resolve().name
    else:
        paths, name = args.images, "panorama"
    if len(paths) < 2:
        sys.exit("need at least two images")
    imgs = load_images(paths)
    print(f"{len(imgs)} images: {', '.join(Path(p).name for p in paths)}")

    methods = list(METHODS) if args.compare else [args.method]
    rows = []
    for m in methods:
        cfg = StitchConfig(
            method=m, blend=args.blend, num_levels=args.levels, gain_compensation=args.gain,
            max_side=args.max_side or None, ratio=args.ratio, ransac_thresh=args.thresh, crop=not args.no_crop
        )
        res = PanoramaStitcher(cfg).stitch(imgs, verbose=args.verbose)
        if not res.ok:
            print(f"[{m}] failed: {res.message}")
            rows.append((m, "failed", "", "", "", f"{res.timings.get('features', 0) + res.timings.get('matching', 0):.1f}"))
            continue
        if args.out and not args.compare:
            out = args.out
        else:
            stem = args.out.with_suffix("") if args.out else (PANO_RESULT_DIR / name / name if args.scene else Path(name))
            out = Path(f"{stem}_{m}.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), res.panorama, [cv2.IMWRITE_JPEG_QUALITY, 92] if out.suffix.lower() in (".jpg", ".jpeg") else [])
        tree_inl = sum(w for _, _, w in res.tree)
        rmse = sum(res.pairs[(i, j)].rmse for i, j, _ in res.tree) / len(res.tree)
        dropped = f", dropped images {res.dropped}" if res.dropped else ""
        print(f"[{m}] panorama {res.panorama.shape[1]}x{res.panorama.shape[0]} from images {res.used}{dropped} "
              f"(reference: {res.root}) -> {out}")
        rows.append((m, f"{len(res.used)}/{len(imgs)}", tree_inl, f"{rmse:.2f}", f"{res.timings['total']:.1f}", ""))

    if args.compare:
        print(f"\n{'method':8s} {'images':>7s} {'tree inliers':>13s} {'RMSE px':>8s} {'time s':>7s}")
        for m, used, inl, rmse, t, _ in rows:
            print(f"{m:8s} {used:>7s} {inl!s:>13s} {rmse:>8s} {t:>7s}")


if __name__ == "__main__":
    main()
