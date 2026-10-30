Real scenes, multi-band blending, default parameters.  inliers = RANSAC inliers on the pairs of the spanning tree, ratio = inliers / putative matches, RMSE = RMS reprojection error of the inliers (px), NCC / MAE = agreement of the warped images inside their overlaps (high-pass NCC: 1 is perfect; MAE in grey levels).

| scene | extractor | keypoints / image | inliers | ratio | RMSE px | NCC high-pass | MAE | time s |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aqueduct | SIFT | 4000 | 1688 | 0.99 | 0.24 | 0.950 | 4.4 | 1.9 |
| aqueduct | ORB | 4000 | 1062 | 0.95 | 0.97 | 0.947 | 4.5 | 1.5 |
| aqueduct | Harris + MOPS | 4000 | 1472 | 0.99 | 0.55 | 0.950 | 4.4 | 2.7 |
| boat | SIFT | 1436 | 1115 | 0.88 | 0.52 | 0.763 | 10.6 | 13.1 |
| boat | ORB | 3833 | 2044 | 0.89 | 1.09 | 0.743 | 10.4 | 13.3 |
| boat | Harris + MOPS | 1889 | 1828 | 0.90 | 0.77 | 0.735 | 10.2 | 12.7 |
| cathedral | SIFT | 3778 | 1974 | 0.89 | 1.12 | 0.739 | 8.7 | 3.0 |
| cathedral | ORB | 3828 | 2520 | 0.96 | 1.15 | 0.776 | 9.5 | 2.8 |
| cathedral | Harris + MOPS | 1635 | 1324 | 0.92 | 1.20 | 0.734 | 8.9 | 2.5 |
| mountain | SIFT | 1924 | 164 | 0.82 | 1.06 | 0.788 | 19.9 | 1.8 |
| mountain | ORB | 3840 | 307 | 0.89 | 1.29 | 0.761 | 20.0 | 1.9 |
| mountain | Harris + MOPS | 1454 | 269 | 0.92 | 1.15 | 0.779 | 19.9 | 1.6 |
| newspaper | SIFT | 4000 | 3305 | 0.89 | 0.32 | 0.927 | 7.4 | 6.3 |
| newspaper | ORB | 4000 | 2485 | 0.90 | 0.89 | 0.905 | 7.9 | 5.4 |
| newspaper | Harris + MOPS | 4000 | 4310 | 0.98 | 0.49 | 0.925 | 7.4 | 8.2 |

Mean over the scenes:

| extractor | keypoints / image | ratio | RMSE px | NCC high-pass | MAE | time s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| SIFT | 3028 | 0.89 | 0.65 | 0.833 | 10.2 | 5.2 |
| ORB | 3900 | 0.92 | 1.08 | 0.826 | 10.5 | 5.0 |
| Harris + MOPS | 2596 | 0.94 | 0.83 | 0.825 | 10.2 | 5.6 |

Time per stage (ms, mean over all scenes): SIFT: features/image 144, matching/pair 91, RANSAC/pair 43; ORB: features/image 18, matching/pair 151, RANSAC/pair 29; Harris + MOPS: features/image 297, matching/pair 70, RANSAC/pair 30
