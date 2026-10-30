# Data of Part 2 (panorama stitching)

```
input/<scene>/   photographs of one scene (any order, two or more)
result/          everything the scripts write: panoramas, CSV / markdown tables, figures
```

## Sample photographs

The five scenes in `input/` are the stitching test images of the OpenCV project
(<https://github.com/opencv/opencv_extra>, folder `testdata/stitching/`). They were copied unchanged, the files are
byte-identical to the originals, and only grouped and renamed per scene:

| scene | photographs | original files in `testdata/stitching/` |
|---|---:|---|
| `boat` | 4 | `boat1.jpg` – `boat4.jpg` |
| `newspaper` | 4 | `newspaper1.jpg` – `newspaper4.jpg` |
| `cathedral` | 3 | `a1.png`, `a2.png`, `a3.png` |
| `mountain` | 2 | `b1.png`, `b2.png` |
| `aqueduct` | 2 | `s1.jpg`, `s2.jpg` |

They are used here only to test the code. Their copyright stays with the original authors; for any other use, replace
them with your own photographs. (The first photograph of `cathedral` and of `mountain` is a greyscale image, the others are
in colour, a property of the original data, discussed in `notebooks/demo_panorama.ipynb`.)

## Your own photographs

Create `input/<name>/` with two or more overlapping photographs taken from one position with the camera turned between the
shots, then

```bash
python run_panorama.py --scene <name> --compare        # three extractors, one comparison table
python scripts/run_scenes.py --scenes <name>           # + CSV rows, keypoint / match / blending figures
```

## What is in `result/`

| file | content |
|---|---|
| `<scene>/<scene>_<extractor>.jpg` | the panoramas (default settings, multi-band blending) |
| `scenes.csv`, `pairs.csv` | one row per (scene, extractor) and per (scene, extractor, image pair): keypoints, matches, inliers, RMSE, overlap agreement, run time of every stage |
| `scenes_summary.md` | the same as a table, with the means over the scenes |
| `figures/keypoints_*`, `matches_*`, `panorama_*`, `blend_*` | per scene: where the detectors fire, the same pair matched by the three extractors, the three panoramas, the blending modes at a seam |
| `figures/scenes_metrics.png`, `runtime.png` | the real scenes as charts |
| `scenes_data.md` | (`scripts/describe_scenes.py`) sizes of the photographs before and after resizing, greyscale photographs, overlap of every pair |
| `overlap_bands_<scene>_<extractor>.md` | (`scripts/analyse_overlap_bands.py`) agreement of two warped images in six bands of their overlap, with and without a shift of one image, and where the inliers lie |
| `chain_drift.md` | (`scripts/analyse_chain_drift.py`) disagreement between the chained homographies and independently estimated pairs |
| `figures/sweep_*.png`, `synthetic/` | controlled study with ground truth: trials (`trials.csv`), summary per level (`summary.csv`, `summary.md`) and charts |

`scripts/make_figures.py` rebuilds all charts and markdown tables from the CSV files. The run times in the tables vary by tens of per
cent between runs on a shared machine; the other columns are reproducible (re-running `scripts/run_scenes.py` gives identical numbers).
