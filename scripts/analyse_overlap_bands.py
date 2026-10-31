#!/usr/bin/env python
"""
Analyzes image agreement in horizontal bands of their overlapping region.

Why is the agreement of two warped images low in some part of their overlap? 
Two different reasons look alike in a single number (the high-pass NCC) and need different remedies:
  * the CONTENT changed between the two exposures (water, ice, leaves, people) -> no shift of the image restores
    the correlation;
  * the REGISTRATION is off there (a homography is only as good as the matches that determined it and is
    extrapolated into regions the matches do not cover) -> a small shift restores it.

For every pair in the spanning tree, this script cuts the overlap into six horizontal bands (as in
`panorama.metrics.overlap_ncc_bands`). For each band, it prints:
    pixels        size of the band
    sigma A / B   standard deviation of the high-pass luminance of the two images (grey levels): how much structure
                  there is to correlate
    NCC           high-pass NCC with the images as warped (the value of `overlap_ncc_bands`)
    best NCC      the best NCC found when image B is shifted by up to +-R pixels, and the shift (dx, dy) at which it is
                  found: B(x + dx, y + dy) is compared with A(x, y)

Additionally, per pair, it shows how the RANSAC inliers are spread over the height of the overlap 
(percentiles of their rows on the canvas). Output is sent to stdout and saved to 
`data/panorama/result/overlap_bands_<scene>_<method>.md`.

Example:
    $ python scripts/analyse_overlap_bands.py                       # boat, SIFT  (~1.5 min)
    $ python scripts/analyse_overlap_bands.py --scene newspaper --method orb --radius 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from panorama.config import PANO_INPUT_DIR, PANO_RESULT_DIR, StitchConfig
from panorama.homography import apply_homography
from panorama.metrics import overlap_ncc_bands
from panorama.stitcher import PanoramaStitcher, list_images, load_images

ROOT = Path(__file__).resolve().parent.parent
N_BANDS, ERODE = 6, 9


def luminance(img: np.ndarray) -> np.ndarray:
    """
    Converts a BGR image to luminance (Y channel).

    Follows the same conversion as `panorama.metrics.overlap_photometric_error`.

    Args:
        img (np.ndarray): The input BGR image.

    Returns:
        np.ndarray: The luminance channel.
    """
    return img @ np.array([0.114, 0.587, 0.299], dtype=np.float32)          # BGR -> Y, as in panorama.metrics


def highpass(img: np.ndarray) -> np.ndarray:
    """
    Applies a high-pass filter (Difference of Gaussians) to the image.

    Args:
        img (np.ndarray): The input BGR image.

    Returns:
        np.ndarray: The high-pass filtered luminance image.
    """
    g = luminance(img)
    return g - cv2.GaussianBlur(g, (0, 0), 3.0)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the zero-mean normalized cross-correlation between two arrays.

    Args:
        a (np.ndarray): The first array.
        b (np.ndarray): The second array.

    Returns:
        float: The NCC value, or NaN if the standard deviation is too small.
    """
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def best_shift(ga: np.ndarray, gb: np.ndarray, sel: np.ndarray, radius: int) -> tuple[float, int, int]:
    """
    Finds the shift of B that maximizes NCC against A within a given radius.

    Searches over integer shifts |dx|, |dy| <= radius for the pixels defined by `sel`.
    B is sampled at p + d and compared with A at p.

    Args:
        ga (np.ndarray): High-pass image A.
        gb (np.ndarray): High-pass image B.
        sel (np.ndarray): Boolean mask of the pixels to evaluate in the overlap band.
        radius (int): Maximum shift (in pixels) to search in both x and y directions.

    Returns:
        tuple[float, int, int]: A tuple containing the best NCC found, the corresponding 
            dx shift, and the dy shift.
    """
    ys, xs = np.nonzero(sel)
    h, w = gb.shape
    best = (ncc(ga[ys, xs], gb[ys, xs]), 0, 0)
    for dy in range(-radius, radius + 1):
        y2 = ys + dy
        for dx in range(-radius, radius + 1):
            x2 = xs + dx
            ok = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
            if ok.sum() < 500:
                continue
            v = ncc(ga[ys[ok], xs[ok]], gb[y2[ok], x2[ok]])
            if v > best[0]:
                best = (v, dx, dy)
    return best


def main() -> None:
    """Parses arguments, runs the overlap band analysis, and outputs the results."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="boat")
    ap.add_argument("--method", default="sift", choices=["sift", "orb", "harris"])
    ap.add_argument("--radius", type=int, default=12, help="largest shift searched (px)")
    ap.add_argument("--input", type=Path, default=PANO_INPUT_DIR)
    args = ap.parse_args()

    images = load_images(list_images(args.input / args.scene))
    res = PanoramaStitcher(StitchConfig(method=args.method)).stitch(images, keep_intermediate=True)
    if not res.ok:
        sys.exit(f"stitching failed: {res.message}")
    pos = {k: a for a, k in enumerate(res.used)}

    out = [f"# Overlap bands of '{args.scene}' ({args.method}), shifts searched: +-{args.radius} px", ""]
    for i, j, _ in res.tree:
        a, b = pos[i], pos[j]
        ov = res.masks[a] & res.masks[b]
        ov = cv2.erode(ov.astype(np.uint8), np.ones((2 * ERODE + 1, 2 * ERODE + 1), np.uint8)).astype(bool)
        rows = np.flatnonzero(ov.any(axis=1))
        y = np.arange(ov.shape[0])[:, None]
        edges = np.linspace(rows[0], rows[-1] + 1, N_BANDS + 1)
        ga, gb = highpass(res.warped[a]), highpass(res.warped[b])
        library = overlap_ncc_bands(res.warped[a], res.masks[a], res.warped[b], res.masks[b], N_BANDS, ERODE)

        info = res.pairs[(min(i, j), max(i, j))]
        kp = res.features[info.i].kpts[info.matches.idx1[info.inliers]]
        cy = apply_homography(res.homographies[info.i], kp)[:, 1]
        p5, p50, p95 = np.percentile(cy, [5, 50, 95])
        frac = lambda v: 100.0 * (v - rows[0]) / (rows[-1] + 1 - rows[0])
        out += [f"## pair {i}-{j}: {info.n_inliers} inliers, overlap rows {rows[0]}..{rows[-1]} of the canvas", "",
                f"RANSAC inliers: rows at the 5th / 50th / 95th percentile = {p5:.0f} / {p50:.0f} / {p95:.0f} "
                f"(= {frac(p5):.0f} % / {frac(p50):.0f} % / {frac(p95):.0f} % of the height of the overlap, from the top); "
                f"{100.0 * (p95 - p5) / (rows[-1] + 1 - rows[0]):.0f} % of the height lies between the 5th and the 95th.", "",
                "| band | pixels | sigma A | sigma B | NCC | best NCC | shift (dx, dy) px |",
                "|--:|--:|--:|--:|--:|--:|:--:|"]
        for k in range(N_BANDS):
            sel = ov & (y >= edges[k]) & (y < edges[k + 1])
            n = int(sel.sum())
            if n < 500:
                out.append(f"| {k + 1} | {n} | | | nan | | |")
                continue
            z = ncc(ga[sel], gb[sel])
            assert abs(z - library[k]) < 1e-5, "disagrees with panorama.metrics.overlap_ncc_bands"
            bn, dx, dy = best_shift(ga, gb, sel, args.radius)
            out.append(f"| {k + 1} | {n} | {255 * ga[sel].std():.1f} | {255 * gb[sel].std():.1f} | {z:.2f} | {bn:.2f} | ({dx:+d}, {dy:+d}) |")
        out.append("")
        print("\n".join(out[-(N_BANDS + 6):]), flush=True)

    target = PANO_RESULT_DIR / f"overlap_bands_{args.scene}_{args.method}.md"
    target.write_text("\n".join(out), encoding="utf-8")
    print(f"written {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()