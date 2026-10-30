# Drift of the chained homographies against independently estimated pairs

| scene | extractor | pair | tree edges between them | share of image i inside j | inliers | RMS drift (px) | max drift (px) |
|:--|:--|:--:|--:|--:|--:|--:|--:|
| boat | sift | 0-2 | 2 | 32 % | 256 | 0.65 | 1.23 |
| boat | sift | 1-3 | 2 | 14 % | 30 | 12.85 | 28.09 |
| boat | orb | 0-2 | 2 | 31 % | 270 | 1.88 | 5.52 |
| boat | harris | 0-2 | 2 | 31 % | 350 | 0.69 | 1.83 |
| boat | harris | 1-3 | 2 | 14 % | 31 | 1.84 | 3.76 |
| cathedral | sift | 0-2 | 2 | 45 % | 585 | 2.83 | 5.94 |
| cathedral | orb | 0-2 | 2 | 45 % | 1183 | 0.62 | 2.13 |
| cathedral | harris | 0-2 | 2 | 44 % | 373 | 3.72 | 12.19 |
| newspaper | sift | 1-3 | 2 | 35 % | 587 | 0.08 | 0.18 |
| newspaper | orb | 1-3 | 2 | 35 % | 579 | 0.26 | 0.69 |
| newspaper | harris | 1-3 | 2 | 35 % | 714 | 0.10 | 0.28 |
