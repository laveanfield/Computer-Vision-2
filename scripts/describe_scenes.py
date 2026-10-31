#!/usr/bin/env python
"""
Describes the photographs of every scene.

Outputs the sizes before and after the resizing of the pipeline, identifies which 
photographs are greyscale, and calculates how much every pair of photographs overlaps 
once the panorama is built (using SIFT with default parameters).

Example:
    Run on every folder of data/panorama/input/ (~30 s):
        $ python scripts/describe_scenes.py
        
    Run on specific scenes with a custom input directory:
        $ python scripts/describe_scenes.py --scenes boat --input path/to/my_scenes

The overlap of the pair (i, j) is the share of image i that falls inside image j, and 
the share of image j inside image i, both computed from the homographies of the final 
panorama (chained through the spanning tree, so a pair that is not an edge of the tree 
has no homography of its own here). Edges of the spanning tree are marked with an 
asterisk (*). 

Output is sent to stdout and saved to `data/panorama/result/scenes_data.md`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from panorama.config import PANO_INPUT_DIR, PANO_RESULT_DIR, StitchConfig
from panorama.stitcher import PanoramaStitcher, list_images, load_images
from panorama.synthetic import overlap_fraction

ROOT = Path(__file__).resolve().parent.parent


def is_grey(im: np.ndarray) -> bool:
    """
    Determines if an image is grayscale.

    Args:
        im (np.ndarray): The input image array.

    Returns:
        bool: True if the image is 2D or if all three color channels are identical, 
            False otherwise.
    """
    return im.ndim == 2 or bool((im[..., 0] == im[..., 1]).all() and (im[..., 1] == im[..., 2]).all())


def sizes(images: list[np.ndarray]) -> str:
    """
    Formats the distinct sizes of a list of images into a readable string.

    Returns distinct sizes as 'width x height'. If there are multiple different 
    sizes, the count for each size is appended in parentheses.

    Args:
        images (list[np.ndarray]): A list of image arrays.

    Returns:
        str: A comma-separated string of distinct image sizes and their counts.
    """
    seen: dict[tuple[int, int], int] = {}
    for im in images:
        seen[(im.shape[1], im.shape[0])] = seen.get((im.shape[1], im.shape[0]), 0) + 1
    return ", ".join(f"{w} x {h}" + (f" ({n})" if len(seen) > 1 else "") for (w, h), n in seen.items())


def main() -> None:
    """
    Parses arguments and runs the scene description analysis.
    
    Loads images for each scene, builds a panorama using SIFT, calculates image overlaps, 
    identifies grayscale images, and writes a summarized Markdown table to the results directory.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=None, help="default: every folder of the input directory")
    ap.add_argument("--input", type=Path, default=PANO_INPUT_DIR)
    args = ap.parse_args()
    scenes = args.scenes or sorted(p.name for p in args.input.iterdir() if p.is_dir())

    out = ["# Scenes: sizes, greyscale photographs, overlaps (SIFT, default parameters)", "",
           "| scene | images | size (width x height) | after resizing | greyscale photographs | overlap of every pair (% of i in j / j in i), * = spanning-tree edge |",
           "|:--|--:|:--|:--|:--|:--|"]
    for scene in scenes:
        images = load_images(list_images(args.input / scene))
        res = PanoramaStitcher(StitchConfig(method="sift", blend="overwrite")).stitch(images)
        grey = [str(k) for k, im in enumerate(images) if is_grey(im)] or ["none"]
        tree = {(min(i, j), max(i, j)) for i, j, _ in res.tree}
        cells = []
        if res.ok:
            used = sorted(res.used)
            for a, i in enumerate(used):
                for j in used[a + 1:]:
                    H = np.linalg.inv(res.homographies[j]) @ res.homographies[i]
                    ij = overlap_fraction(H, res.images[i].shape, res.images[j].shape)
                    ji = overlap_fraction(np.linalg.inv(H), res.images[j].shape, res.images[i].shape)
                    cells.append(f"({i},{j}){'*' if (i, j) in tree else ''} {100 * ij:.0f}/{100 * ji:.0f}")
        row = (
            f"| {scene} | {len(images)} | {sizes(images)} | {sizes(res.images)} | {', '.join(grey)} | "
            f"{'; '.join(cells) if cells else res.message} |"
        )
        out.append(row)
        print(row, flush=True)

    target = PANO_RESULT_DIR / "scenes_data.md"
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"written {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()