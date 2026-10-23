# Data of Part 1 (image blending)

```
source/   the picture that contains the object to paste
mask/     white where the object is, black elsewhere
target/   the picture the object is pasted onto
result/   what the code writes: the three blended pictures of every image
```

The three folders `source/`, `mask/` and `target/` are matched **by file name**: `source/03.jpg`, `mask/03.jpg` and `target/03.jpg` form one example, and `IMAGE_NAME = "03.jpg"` in `src/image_blending/config.py` selects it. The path of this folder is set by `DATA_ROOT` in `src/image_blending/config.py`.

## Provided examples

| image | source (w × h) | mask (w × h) | target (w × h) |
|---|---|---|---|
| `01.jpg` | 275 × 183 | 275 × 183 | 609 × 369 |
| `02.jpg` | 1200 × 800 | 1200 × 800 | 580 × 326 |
| `03.jpg` | 600 × 425 | 600 × 425 | 612 × 408 |

The source and its mask always have the same size; the target has its own size and the result has the size of the target. The source is placed on the target at an offset and a scale that you choose (see below), so the sizes need not match: the source of `02.jpg` is larger than its target and is scaled by 0.4 (the scale is part of the placement).

> Source of the photographs: Internet
## Your own example

1. Put `source`, `mask` and `target` with the **same file name** (e.g. `04.jpg`) into the three folders.
2. Make the mask in an image editor such as [GIMP](https://www.gimp.org/): select the object (free select or fuzzy select), fill the selection with white on a black background, export.
3. Set `IMAGE_NAME = "04.jpg"` in `src/image_blending/config.py` and run `notebooks/demo.ipynb`. An OpenCV window opens: move the mouse (or click) to position the object, scroll or press `+` / `-` to scale it, Enter to confirm. To skip the window, set `USE_MANUAL_OFFSET = True` and add `"04.jpg": [dy, dx, scale]` to `MANUAL_OFFSETS`.

## What is in `result/`

| file | content |
|---|---|
| `<image>_naive.jpg` | naive copy, $C = \alpha F + (1-\alpha)B$ |
| `<image>_laplacian.jpg` | Laplacian pyramid blending (`NUM_LEVELS = 3`) |
| `<image>_poisson.jpg` | Poisson blending (`GRAD_MIX = False`) |
| `<image>.jpg` | one more picture per example; it is not written by the notebook or the scripts of this repository |

The three `_naive`, `_laplacian` and `_poisson` files are written by `notebooks/demo_image_blending.ipynb` as JPEG (quality 95), overwriting existing files with the same name.