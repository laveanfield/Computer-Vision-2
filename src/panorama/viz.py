"""
Plotting helpers (matplotlib) for the panorama experiments.

Visual language:
    - Every feature extractor owns one fixed colour *and* one marker shape, in every figure of the project:
        SIFT = blue circle      ORB = orange square      Harris + MOPS = aqua triangle
      The three hues are the first three slots of a validated categorical palette (all-pairs colour-vision-deficiency
      separation dE >= 9, normal-vision dE >= 24). Aqua has only 2.7:1 contrast on the light surface, therefore identity is
      never carried by colour alone: marker shapes, an always-present legend and the CSV / markdown tables written next to
      every figure ("table view") carry it as well.
    - Thin marks: 2 px lines, >= 8 px markers with a 2 px surface ring, bars <= 24 px, hairline solid grid, quiet axes.
    - Text never wears a data colour (text tokens only); the coloured mark next to the text identifies the series.
    - No dual axes, no dashed grid lines, a number on a point only where it carries the story.
"""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import cv2
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from .blend import blend_images, compute_gains
from .features import METHODS
from .metrics import overlap_photometric_error

#=================================================
# Design tokens
#=================================================

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"        # primary / secondary / muted text
GRID, AXIS = "#e1e0d9", "#c3c2b7"                         # hairline grid / axis line

SERIES = {
    "sift":   dict(color="#2a78d6", marker="o", label="SIFT"),
    "orb":    dict(color="#eb6834", marker="s", label="ORB"),
    "harris": dict(color="#1baf7a", marker="^", label="Harris + MOPS"),
}
NEUTRAL = MUTED                                           # de-emphasis colour (outliers, context)


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    """Converts a hex color string to a BGR integer tuple.

    Args:
        h (str): Hex color string, e.g., '#2a78d6'.

    Returns:
        tuple[int, int, int]: A tuple containing the (Blue, Green, Red) integer values.
    """
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def apply_style() -> None:
    """Installs the project-wide matplotlib style (idempotent)."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 100, "savefig.dpi": 150,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 5,
        "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 7.5,
        "image.interpolation": "nearest",
    })


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Saves and closes a matplotlib figure.

    Uses `.jpg` or `.jpeg` for photo-heavy figures to keep files small; otherwise uses `.png` for charts.

    Args:
        fig (plt.Figure): The matplotlib figure object.
        path (str | Path): The destination file path.
        dpi (int, optional): The resolution in dots per inch. Defaults to 150.

    Returns:
        Path: The absolute or relative Path object where the figure was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"pil_kwargs": {"quality": 88, "optimize": True}} if path.suffix.lower() in (".jpg", ".jpeg") else {}
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE, **kw)
    plt.close(fig)
    return path


def method_handles(methods: tuple[str, ...] = METHODS, with_line: bool = True) -> list[Line2D | Patch]:
    """Generates legend handles for the defined feature extraction methods.

    Args:
        methods (tuple[str, ...], optional): A collection of method names. Defaults to METHODS.
        with_line (bool, optional): If True, returns `Line2D` handles showing the line and marker. 
            If False, returns `Patch` objects (solid colored rectangles). Defaults to True.

    Returns:
        list[Line2D | Patch]: A list of matplotlib legend handles.
    """
    if with_line:
        return [Line2D([0], [0], color=SERIES[m]["color"], marker=SERIES[m]["marker"], markersize=7.5,
                       markeredgecolor=SURFACE, markeredgewidth=1.5, linewidth=2.0, label=SERIES[m]["label"])
                for m in methods]
    return [Patch(facecolor=SERIES[m]["color"], edgecolor="none", label=SERIES[m]["label"]) for m in methods]


def _rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Converts an image from BGR color space to RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def show_image(ax: plt.Axes, img_bgr: np.ndarray, title: str | None = None) -> None:
    """Displays a BGR image on a given matplotlib Axes object.

    Args:
        ax (plt.Axes): The matplotlib Axes to draw on.
        img_bgr (np.ndarray): The image array in BGR format.
        title (str | None, optional): The title for the subplot. Defaults to None.
    """
    ax.imshow(_rgb(img_bgr))
    ax.set_axis_off()
    ax.grid(False)
    if title:
        ax.set_title(title, loc="left")


#=================================================
# Keypoints, matches, layout
#=================================================

def draw_keypoints(img_bgr: np.ndarray, feats, method: str, n_circles: int = 100, seed: int = 0) -> np.ndarray:
    """Draws keypoints as dots on an image, adding scale circles to a random subset.

    Args:
        img_bgr (np.ndarray): The background image array in BGR format.
        feats: An object containing `.kpts` (keypoint coordinates) and `.scale` (keypoint scales).
        method (str): The method name key to determine the drawing color from `SERIES`.
        n_circles (int, optional): The number of random keypoints to draw scale circles for. Defaults to 100.
        seed (int, optional): Random seed for selecting the subset. Defaults to 0.

    Returns:
        np.ndarray: The resulting image array with keypoints drawn.
    """
    out = (img_bgr.astype(np.float32) * 0.78 + 255 * 0.22).astype(np.uint8)         # slightly washed out -> marks pop
    col = hex_to_bgr(SERIES[method]["color"])
    for x, y in feats.kpts:
        cv2.circle(out, (int(round(x)), int(round(y))), 2, col, -1, cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    for k in rng.choice(len(feats), min(n_circles, len(feats)), replace=False):
        x, y = feats.kpts[k]
        cv2.circle(out, (int(round(x)), int(round(y))), int(np.clip(feats.scale[k] / 2.0, 4, 60)), col, 1, cv2.LINE_AA)
    return out


def keypoint_figure(img_bgr: np.ndarray, feats_by_method: dict, n_circles: int = 100, width: float = 12.0) -> plt.Figure:
    """Creates a figure comparing detected keypoints across different extractors on a single image.

    Args:
        img_bgr (np.ndarray): The background BGR image.
        feats_by_method (dict): A mapping of method names to their respective features.
        n_circles (int, optional): The number of scale circles to draw per method. Defaults to 100.
        width (float, optional): The total width of the matplotlib figure in inches. Defaults to 12.0.

    Returns:
        plt.Figure: The constructed matplotlib figure.
    """
    apply_style()
    methods = [m for m in METHODS if m in feats_by_method]
    h, w = img_bgr.shape[:2]
    fig, axes = plt.subplots(1, len(methods), figsize=(width, width / len(methods) * h / w + 0.5))
    for ax, m in zip(np.atleast_1d(axes), methods):
        f = feats_by_method[m]
        show_image(ax, draw_keypoints(img_bgr, f, m, n_circles), f"{SERIES[m]['label']}  -  {len(f)} keypoints")
    fig.text(0.01, -0.01, f"dot = keypoint, circle = scale of {n_circles} randomly chosen keypoints", color=INK2, fontsize=9, va="top")
    fig.tight_layout(w_pad=1.0)
    return fig


def _montage(img_a: np.ndarray, img_b: np.ndarray, gap: int = 12) -> tuple[np.ndarray, int]:
    """Combines two images side-by-side into a single montage image with a gap.

    Args:
        img_a (np.ndarray): The left image.
        img_b (np.ndarray): The right image.
        gap (int, optional): The width of the gap between the images in pixels. Defaults to 12.

    Returns:
        tuple[np.ndarray, int]: A tuple containing:
            - canvas (np.ndarray): The resulting montage image.
            - offset (int): The x-coordinate offset for points mapped to the right image.
    """
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    h = max(ha, hb)
    canvas = np.full((h, wa + gap + wb, 3), (251, 252, 252), np.uint8)
    canvas[:ha, :wa] = img_a
    canvas[:hb, wa + gap:] = img_b
    return canvas, wa + gap


def draw_matches(ax: plt.Axes, img_a: np.ndarray, img_b: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray, inliers: np.ndarray, method: str, max_inliers: int = 80, max_outliers: int = 50,
                 seed: int = 0, title: str | None = None) -> None:
    """Draws lines connecting matched feature points between two side-by-side images.

    RANSAC inliers are drawn in the extractor's designated color; rejected matches are 
    drawn in a neutral grey.

    Args:
        ax (plt.Axes): The matplotlib Axes to draw on.
        img_a (np.ndarray): The left source image.
        img_b (np.ndarray): The right destination image.
        pts_a (np.ndarray): Array of [x, y] coordinates in the left image.
        pts_b (np.ndarray): Array of [x, y] coordinates in the right image.
        inliers (np.ndarray): Boolean mask indicating which matches are RANSAC inliers.
        method (str): The method name key for styling from `SERIES`.
        max_inliers (int, optional): The maximum number of inlier lines to draw. Defaults to 80.
        max_outliers (int, optional): The maximum number of outlier lines to draw. Defaults to 50.
        seed (int, optional): Random seed for sampling the lines to draw. Defaults to 0.
        title (str | None, optional): The title for the subplot. Defaults to None.
    """
    rng = np.random.default_rng(seed)
    canvas, dx = _montage(img_a, img_b)
    ax.imshow(_rgb(canvas))
    ax.set_axis_off()
    ax.grid(False)
    inliers = np.asarray(inliers, dtype=bool)
    pa, pb = np.asarray(pts_a, dtype=np.float64), np.asarray(pts_b, dtype=np.float64)

    def pick(mask: np.ndarray, n: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        return rng.choice(idx, n, replace=False) if len(idx) > n else idx

    out_idx, in_idx = pick(~inliers, max_outliers), pick(inliers, max_inliers)
    for idx, color, lw, alpha, z in ((out_idx, NEUTRAL, 0.7, 0.55, 2), (in_idx, SERIES[method]["color"], 1.0, 0.85, 3)):
        if len(idx) == 0:
            continue
        segs = np.stack([np.stack([pa[idx, 0], pa[idx, 1]], axis=1),
                         np.stack([pb[idx, 0] + dx, pb[idx, 1]], axis=1)], axis=1)
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, zorder=z))
        ax.scatter(pa[idx, 0], pa[idx, 1], s=7, color=color, zorder=z + 1, linewidths=0)
        ax.scatter(pb[idx, 0] + dx, pb[idx, 1], s=7, color=color, zorder=z + 1, linewidths=0)
    if title:
        ax.set_title(title, loc="left")


def matches_figure(img_a: np.ndarray, img_b: np.ndarray, per_method: dict, width: float = 11.0, title: str | None = None) -> plt.Figure:
    """Creates a figure with stacked rows showing feature matches for different extractors.

    Args:
        img_a (np.ndarray): The left source image.
        img_b (np.ndarray): The right destination image.
        per_method (dict): A dictionary where keys are method names and values are dicts containing 
            'pts_a', 'pts_b', 'inliers', 'n_matches', 'n_inliers', and 'rmse'.
        width (float, optional): The total width of the figure in inches. Defaults to 11.0.
        title (str | None, optional): The main title for the entire figure. Defaults to None.

    Returns:
        plt.Figure: The constructed matplotlib figure.
    """
    apply_style()
    methods = [m for m in METHODS if m in per_method]
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    row_h = width * max(ha, hb) / (wa + wb + 12)
    fig, axes = plt.subplots(len(methods), 1, figsize=(width, row_h * len(methods) + 0.4 * len(methods)))
    for ax, m in zip(np.atleast_1d(axes), methods):
        d = per_method[m]
        t = (f"{SERIES[m]['label']}  -  {d['n_matches']} putative matches, {d['n_inliers']} RANSAC inliers "
             f"({100.0 * d['n_inliers'] / max(d['n_matches'], 1):.0f} %), RMSE {d['rmse']:.2f} px")
        draw_matches(ax, img_a, img_b, d["pts_a"], d["pts_b"], d["inliers"], m, title=t)
    legend = [Line2D([0], [0], color=NEUTRAL, lw=1.5, label="rejected by RANSAC"),
              Line2D([0], [0], color=INK2, lw=2.0, label="RANSAC inlier (colour = extractor)")]
    fig.legend(handles=legend, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 0.965 if title else 1), h_pad=1.2)
    if title:
        fig.suptitle(title, x=0.01, ha="left", y=0.995, fontsize=11.5, fontweight="bold")
    return fig


def draw_layout(ax: plt.Axes, res, title: str | None = None) -> None:
    """Draws the footprints of the warped images and the spanning tree of the overlap graph on the canvas.

    Args:
        ax (plt.Axes): The matplotlib Axes to draw on.
        res: The stitching result object containing `.canvas_shape`, `.used`, `.images`, 
            `.homographies`, `.tree`, and `.root`.
        title (str | None, optional): The title for the subplot. Defaults to None.
    """
    pano_h, pano_w = res.canvas_shape
    ax.set_xlim(0, pano_w)
    ax.set_ylim(pano_h, 0)
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    centres = {}
    for k in res.used:
        h, w = res.images[k].shape[:2]
        corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=float)
        q = corners @ res.homographies[k].T
        q = q[:, :2] / q[:, 2:3]
        ax.add_patch(Polygon(q, closed=True, facecolor=INK2, alpha=0.10, edgecolor=INK2, linewidth=1.2))
        centres[k] = q.mean(axis=0)
    for i, j, _ in res.tree:
        (x0, y0), (x1, y1) = centres[i], centres[j]
        ax.plot([x0, x1], [y0, y1], color=INK, linewidth=1.5, zorder=3)
    for k, (x, y) in centres.items():
        ref = k == res.root
        ax.plot(x, y, marker="o", markersize=17, markerfacecolor=SURFACE, markeredgecolor=INK,
                markeredgewidth=3.0 if ref else 1.2, zorder=5)
        ax.text(x, y, str(k), ha="center", va="center", color=INK, fontsize=9, fontweight="bold", zorder=6)
    edges = ", ".join(f"{i}-{j}: {n} inliers" for i, j, n in res.tree)
    ax.text(0.0, -0.015, textwrap.fill(f"tree edges  {edges}\nthick ring = reference image", 52, replace_whitespace=False),
            transform=ax.transAxes, ha="left", va="top", color=INK2, fontsize=8.5)
    if title:
        ax.set_title(title, loc="left")


BLEND_VARIANTS = (("overwrite", "overwrite", False), ("average", "average", False), ("feather", "feather", False),
                  ("multiband", "multiband", False), ("multiband + gain", "multiband", True))


def blend_comparison_figure(res, window: tuple[int, int] = (420, 280), margin: int = 48, variants: tuple = BLEND_VARIANTS) -> plt.Figure:
    """Creates a figure comparing different blending modes on a cropped overlap region.

    Extracts a crop around the most overlapping pair of a stitched scene and applies 
    various blending methods to visualize the differences.

    Args:
        res: The panorama result object containing intermediate warped images and masks 
            (must be created with `keep_intermediate=True`).
        window (tuple[int, int], optional): The (width, height) of the crop window. Defaults to (420, 280).
        margin (int, optional): The extra margin added during blending to prevent edge artifacts. Defaults to 48.
        variants (tuple, optional): A collection of tuples defining (label, mode, use_gain). Defaults to BLEND_VARIANTS.

    Returns:
        plt.Figure: The constructed matplotlib figure.
    """
    apply_style()
    pos = {k: a for a, k in enumerate(res.used)}
    best, best_n = None, -1
    for i, j, _ in res.tree:
        n = int((res.masks[pos[i]] & res.masks[pos[j]]).sum())
        if n > best_n:
            best, best_n = (i, j), n
    ia, ib = pos[best[0]], pos[best[1]]
    ov = res.masks[ia] & res.masks[ib]
    ys, xs = np.nonzero(ov)
    H, W = ov.shape
    cw, ch = min(window[0], W), min(window[1], H)
    x0 = int(np.clip(np.median(xs) - cw // 2, 0, W - cw))
    y0 = int(np.clip(np.median(ys) - ch // 2, 0, H - ch))
    X0, Y0, X1, Y1 = max(x0 - margin, 0), max(y0 - margin, 0), min(x0 + cw + margin, W), min(y0 + ch + margin, H)
    gains = compute_gains(res.warped, res.masks)

    crops, notes = [], []
    for label, mode, use_gain in variants:
        imgs = [w[Y0:Y1, X0:X1] * (gains[a] if use_gain else 1.0) for a, w in enumerate(res.warped)]
        imgs = [np.clip(im, 0, 1) for im in imgs]
        masks = [m[Y0:Y1, X0:X1] for m in res.masks]
        pano, _, _ = blend_images(imgs, masks, mode, res.config.num_levels, res.blend_order, gain_compensation=False)
        crops.append(pano[y0 - Y0:y0 - Y0 + ch, x0 - X0:x0 - X0 + cw])
        wa = np.clip(res.warped[ia] * (gains[ia] if use_gain else 1.0), 0, 1)
        wb = np.clip(res.warped[ib] * (gains[ib] if use_gain else 1.0), 0, 1)
        notes.append(overlap_photometric_error(wa, res.masks[ia], wb, res.masks[ib])["mae"])

    overview, _, _ = blend_images(res.warped, res.masks, "feather", res.config.num_levels, res.blend_order)
    ncols = 3
    nrows = int(np.ceil((len(variants) + 1) / ncols))
    cell_w = 4.1
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, nrows * (cell_w * ch / cw + 0.75)), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    axo = axes[0, 0]
    axo.imshow(_rgb(np.clip(overview * 255 + 0.5, 0, 255).astype(np.uint8)))
    axo.add_patch(plt.Rectangle((x0, y0), cw, ch, fill=False, ec=INK, lw=1.8))
    axo.set_title(f"images {best[0]} and {best[1]}: crop (box)", loc="left", fontsize=10)
    for k, (crop, (label, _, g), mae) in enumerate(zip(crops, variants, notes), start=1):
        ax = axes[k // ncols, k % ncols]
        ax.imshow(_rgb(np.clip(crop * 255 + 0.5, 0, 255).astype(np.uint8)))
        ax.set_title(label, loc="left", fontsize=10)
        ax.text(0.0, -0.03, f"mean |luminance difference| in the overlap: {mae:.1f}", transform=ax.transAxes, ha="left",
                va="top", color=INK2, fontsize=8.5)
    fig.tight_layout(h_pad=1.6, w_pad=0.6)
    return fig


def panorama_figure(panos: dict, captions: dict, width: float = 12.0) -> plt.Figure:
    """Creates a layout combining the generated panoramas from different extractors.

    Panels are placed explicitly: side by side if they are tall, stacked if they are wide.
    A caption consists of a bold title and a subtitle below it.

    Args:
        panos (dict): Dictionary mapping method names to their respective panorama image arrays.
        captions (dict): Dictionary mapping method names to caption strings (can contain "\n" for subtitles).
        width (float, optional): Total width of the figure in inches. Defaults to 12.0.

    Returns:
        plt.Figure: The constructed matplotlib figure.
    """
    apply_style()
    methods = [m for m in METHODS if m in panos]
    aspects = [panos[m].shape[1] / panos[m].shape[0] for m in methods]
    n, gap = len(methods), 0.3                                                              # gap between panels (inches)
    parts = [(captions[m].split("\n", 1) + [""])[:2] for m in methods]          # (title, subtitle)
    tall = np.mean(aspects) < 1.5

    def lines(text: str) -> int:
        return text.count("\n") + 1 if text else 0

    if tall:                                                                                # equal heights, widths ~ aspect ratio
        ph = (width - gap * (n - 1)) / sum(aspects)
        w_in = [a * ph for a in aspects]
        h_in = [ph] * n
    else:                                                                                   # full width, heights ~ 1 / aspect ratio
        w_in = [width] * n
        h_in = [width / a for a in aspects]
    # ~11 bold / ~13 regular characters fit in one inch at these font sizes
    titles = [textwrap.fill(t, max(int(11 * w), 12)) for (t, _), w in zip(parts, w_in)]
    subs = [textwrap.fill(u, max(int(13 * w), 12)) for (_, u), w in zip(parts, w_in)]
    t_h = [0.14 + 0.20 * lines(t) for t in titles]
    s_h = [0.10 + 0.165 * lines(u) for u in subs]
    if tall:
        t_h, s_h = [max(t_h)] * n, [max(s_h)] * n

    if tall:
        total_h = t_h[0] + ph + s_h[0]
        lefts = np.concatenate([[0.0], np.cumsum(np.array(w_in) + gap)[:-1]])
        boxes = [(l, s_h[0], w, ph) for l, w in zip(lefts, w_in)]               # (left, bottom, width, height) in inches
    else:
        total_h = sum(t + h + u for t, h, u in zip(t_h, h_in, s_h)) + gap * (n - 1)
        boxes, top = [], total_h
        for t, h, u in zip(t_h, h_in, s_h):
            boxes.append((0.0, top - t - h, width, h))
            top -= t + h + u + gap
    fig = plt.figure(figsize=(width, total_h))
    for m, (l, b, w, h), title, sub in zip(methods, boxes, titles, subs):
        ax = fig.add_axes([l / width, b / total_h, w / width, h / total_h])
        show_image(ax, panos[m], title)
        if sub:
            ax.text(0.0, -0.012, sub, transform=ax.transAxes, ha="left", va="top", color=INK2, fontsize=8.8)
    return fig


#=================================================
# Tables
#=================================================

def read_csv(path: str | Path) -> list[dict]:
    """Reads a CSV file into a list of dictionaries.

    Numeric cells are converted to floats (empty cells become NaN), while 
    everything else remains a string.

    Args:
        path (str | Path): The path to the CSV file.

    Returns:
        list[dict]: A list of row dictionaries representing the CSV data.
    """
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v) if v != "" else float("nan")
                except ValueError:
                    out[k] = v
            rows.append(out)
    return rows


def markdown_table(headers: list[str], body: list[list], align: str | None = None) -> str:
    """Generates a GitHub-flavored Markdown table from data.

    Args:
        headers (list[str]): A list of string headers for the columns.
        body (list[list]): A list of lists, where each inner list represents a row of cells.
        align (str | None, optional): A string specifying alignment per column using 'l', 'r', or 'c'. 
            Defaults to None (first column left-aligned, rest right-aligned).

    Returns:
        str: The formatted Markdown table string.
    """
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep[a] for a in align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


#=================================================
# Synthetic robustness study
#=================================================

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Computes the 95% Wilson score interval for a binomial proportion.

    This method is well-behaved even for small sample sizes and probabilities near 0 or 1.

    Args:
        k (int): Number of successes.
        n (int): Total number of trials.
        z (float, optional): The z-score corresponding to the desired confidence level. 
            Defaults to 1.96 for 95% confidence.

    Returns:
        tuple[float, float]: A tuple containing the (lower_bound, upper_bound) of the interval.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def aggregate_sweeps(rows: list[dict], tol: float = 3.0) -> dict:
    """Aggregates synthetic sweep data by sweep type and extraction method.

    Calculates success metrics based on a defined tolerance for geometric error.
    - accepted: The pair passed the overlap test.
    - correct: Accepted and the geometric error against ground truth is below `tol` pixels.
    - wrong_accept: Accepted despite incorrect geometry (silent failure).
    - success: correct / n.

    Args:
        rows (list[dict]): The raw result rows containing fields like "sweep", "method", 
            "level", "valid", "gt_err", "n_inliers", "n_kp_a", and "n_kp_b".
        tol (float, optional): The error tolerance in pixels to classify a homography as correct. 
            Defaults to 3.0.

    Returns:
        dict: A nested dictionary structured as {(sweep, method): list_of_level_statistics}. 
            Levels are ordered by appearance (typically easy to hard).
    """
    out: dict = {}
    for r in rows:
        out.setdefault((r["sweep"], r["method"]), {}).setdefault(r["level"], []).append(r)
    agg = {}
    for key, by_level in out.items():
        levels = []
        for level, rs in by_level.items():
            n = len(rs)
            acc = [r for r in rs if r["valid"] == 1]
            err = np.array([r["gt_err"] for r in acc], dtype=float)
            correct = int(np.sum(err < tol))
            lo, hi = wilson_interval(correct, n)
            levels.append(dict(
                level=level, n=n, accepted=len(acc), correct=correct, wrong_accept=len(acc) - correct,
                success=correct / n, ci_lo=lo, ci_hi=hi,
                median_err=float(np.median(err[err < tol])) if correct else float("nan"),
                median_inl=float(np.median([r["n_inliers"] for r in rs])),
                median_kp=float(np.median([0.5 * (r["n_kp_a"] + r["n_kp_b"]) for r in rs]))))
        agg[key] = levels
    return agg


DODGE = {"sift": -0.12, "orb": 0.0, "harris": 0.12}


def _level_label(v: float) -> str:
    """Formats a float level value for axis tick labels."""
    return f"{v:g}"


def sweep_figure(agg: dict, sweeps: dict, metric: str = "success", tol: float = 3.0, ncols: int = 3,
                 title: str | None = None, tick_labels: dict | None = None, xlabels: dict | None = None) -> plt.Figure:
    """Generates a small-multiples grid plot for the difficulty sweep analysis.

    Each panel represents a difficulty factor, and each line corresponds to an extractor.
    Difficulty increases to the right.

    Supported metrics:
      - 'success': Fraction of trials with a correct homography.
      - 'error': Median geometric error (px) of the correct homographies (log axis).
      - 'inliers': Median number of RANSAC inliers (log axis).

    Args:
        agg (dict): The aggregated sweep data from `aggregate_sweeps()`.
        sweeps (dict): Definitions of the sweeps (names to axis labels/levels).
        metric (str, optional): The metric to plot. Defaults to "success".
        tol (float, optional): The tolerance threshold used, included in labels. Defaults to 3.0.
        ncols (int, optional): The number of columns for the subplots grid. Defaults to 3.
        title (str | None, optional): The main figure title. Defaults to None.
        tick_labels (dict | None, optional): Custom tick labels per sweep. Defaults to None.
        xlabels (dict | None, optional): Custom x-axis labels per sweep. Defaults to None.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    apply_style()
    names = [s for s in sweeps if any((s, m) in agg for m in METHODS)]
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.15 * nrows + 0.7), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.set_axis_off()
    for ax, name in zip(axes.ravel(), names):
        xlabel = (xlabels or {}).get(name, sweeps[name][0])
        for m in METHODS:
            lv = agg.get((name, m))
            if not lv:
                continue
            style = SERIES[m]
            x = np.arange(len(lv)) + DODGE[m]                 # tiny horizontal offset so that coinciding series stay visible
            if metric == "success":
                y = np.array([d["success"] for d in lv])
            elif metric == "error":
                y = np.array([d["median_err"] for d in lv])
            else:
                y = np.maximum(np.array([d["median_inl"] for d in lv]), 3.0)     # zero inliers are drawn on the floor
            ax.plot(x, y, color=style["color"], marker=style["marker"], markeredgecolor=SURFACE, markeredgewidth=1.5)
        first = next(iter(agg[(name, m)] for m in METHODS if (name, m) in agg))
        ax.set_xticks(np.arange(len(first)))
        ax.set_xticklabels((tick_labels or {}).get(name) or [_level_label(d["level"]) for d in first])
        ax.set_xlim(-0.3, len(first) - 0.7)
        ax.set_xlabel(xlabel)
        if metric == "success":
            ax.set_ylim(-0.04, 1.04)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0", "25", "50", "75", "100 %"])
        elif metric == "error":
            ax.set_yscale("log")
            ax.set_ylim(0.02, 4.0)
            ax.set_yticks([0.03, 0.1, 0.3, 1, 3])
            ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
            ax.minorticks_off()
        else:
            ax.set_yscale("log")
            ax.set_ylim(2.2, 4000)
            ax.set_yticks([3, 10, 100, 1000])
            ax.set_yticklabels(["<= 3", "10", "100", "1000"])
            ax.minorticks_off()
        ax.set_title(name)
    ylabel = {"success": f"correct homographies (error < {tol:g} px)", "error": "median error (px)",
              "inliers": "median RANSAC inliers"}[metric]
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.legend(handles=method_handles(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=2.5)
    if title:
        fig.suptitle(title, x=0.01, ha="left", y=1.06, fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.6, h_pad=1.6)
    return fig


#=================================================
# Real scenes
#=================================================

def grouped_bars(axes: list[plt.Axes] | np.ndarray, categories: list[str], values: dict, ylabels: list[str], titles: list[str] | None = None,
                 logy: list[bool] | None = None, bar_px: float = 17.0) -> None:
    """Plots grouped bar charts comparing extractors across multiple categories and metrics.

    Args:
        axes (list[plt.Axes] | np.ndarray): The matplotlib Axes to plot on (one per metric).
        categories (list[str]): The labels for the x-axis categories.
        values (dict): A nested dictionary structured as values[metric_index][method] = list_of_values.
        ylabels (list[str]): Y-axis labels for each metric panel.
        titles (list[str] | None, optional): Subplot titles for each panel. Defaults to None.
        logy (list[bool] | None, optional): Boolean list indicating if a panel's y-axis should be logarithmic. Defaults to None.
        bar_px (float, optional): Maximum bar width in pixels. Defaults to 17.0.
    """
    n_methods = len(METHODS)
    x = np.arange(len(categories))
    for pi, ax in enumerate(np.atleast_1d(axes)):
        fig = ax.figure
        ax_w_px = ax.get_position().width * fig.get_figwidth() * fig.dpi
        slot = ax_w_px / max(len(categories), 1)
        w = min(bar_px, 0.82 * slot / n_methods) / slot              # bar width in data units (thin, never fills the slot)
        gap = 0.10 * w                                               # surface gap between neighbouring bars
        for mi, m in enumerate(METHODS):
            off = (mi - (n_methods - 1) / 2) * (w + gap)
            ax.bar(x + off, values[pi][m], width=w, color=SERIES[m]["color"], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel(ylabels[pi])
        if titles:
            ax.set_title(titles[pi])
        if logy and logy[pi]:
            ax.set_yscale("log")
        ax.grid(axis="x", visible=False)