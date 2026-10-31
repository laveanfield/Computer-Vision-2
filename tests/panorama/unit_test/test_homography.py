"""
Unit tests for panorama.homography module following the Input-Output testing model.
"""

from unittest.mock import patch
import numpy as np
import pytest
from panorama.homography import (
    apply_homography,
    reprojection_error,
    symmetric_transfer_error,
    normalize_points,
    dlt_homography,
    dlt_conditioning,
    normalize_homography,
    footprint_ok,
    required_iterations,
    ransac_homography,
    refine_homography,
)


def test_apply_homography_identity():
    """
    [INPUT]:
        - H: 3x3 Identity matrix.
        - pts: (4, 2) coordinates [[0, 0], [10, 0], [10, 20], [0, 20]].
    [OUTPUT]: Projected points must exactly equal input points.
    """
    # --- [1. INPUT] ---
    H = np.eye(3, dtype=np.float64)
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]])

    # --- [2. ACT] ---
    actual_pts = apply_homography(H, pts)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(actual_pts, pts, atol=1e-7, err_msg="Identity H modified points.")


def test_apply_homography_translation_and_scale():
    """
    [INPUT]:
        - H: Scaling by 2.0 and translation by (5.0, 10.0).
        - pts: [[1.0, 2.0], [3.0, 4.0]].
    [OUTPUT]:
        - Projected points: [[2*1 + 5, 2*2 + 10], [2*3 + 5, 2*4 + 10]] = [[7, 14], [11, 18]].
    """
    # --- [1. INPUT] ---
    H = np.array([
        [2.0, 0.0, 5.0],
        [0.0, 2.0, 10.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_pts = np.array([[7.0, 14.0], [11.0, 18.0]])

    # --- [2. ACT] ---
    actual_pts = apply_homography(H, pts)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(actual_pts, expected_pts, atol=1e-7)


def test_apply_homography_point_at_infinity():
    """
    [INPUT]: Homography mapping points to line at infinity (w = 0).
    [OUTPUT]: Homogeneous divisor w ~ 0 must result in NaN values without crashing.
    """
    # --- [1. INPUT] ---
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]  # w = x
    ], dtype=np.float64)
    pts = np.array([[0.0, 5.0]])  # x=0 -> w=0

    # --- [2. ACT] ---
    actual_pts = apply_homography(H, pts)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert np.isnan(actual_pts).all(), "Points with w=0 must evaluate to NaN"


def test_normalize_points_hartley_invariants():
    """
    [INPUT]: Arbitrary 2D point cloud of 5 points.
    [OUTPUT]:
        - Normalized centroid must be at origin (0, 0).
        - Mean Euclidean distance from origin must be sqrt(2) within numerical tolerance.
    """
    # --- [1. INPUT] ---
    pts = np.array([[10.0, 20.0], [50.0, 80.0], [100.0, 120.0], [30.0, 40.0], [70.0, 90.0]])

    # --- [2. ACT] ---
    norm_pts, T = normalize_points(pts)

    # --- [3. OUTPUT ASSERTIONS] ---
    centroid = norm_pts.mean(axis=0)
    np.testing.assert_allclose(centroid, [0.0, 0.0], atol=1e-7, err_msg="Normalized centroid not at (0, 0)")
    mean_dist = np.mean(np.linalg.norm(norm_pts, axis=1))
    assert np.isclose(mean_dist, np.sqrt(2.0)), f"Expected mean distance sqrt(2), got {mean_dist}"


def test_dlt_homography_exact_recovery():
    """
    [INPUT]:
        - 6 non-collinear source points.
        - Destination points transformed by a known ground-truth affine homography H_true.
    [OUTPUT]: DLT must recover H_true up to scale factor (normalized H matching H_true).
    """
    # --- [1. INPUT] ---
    src = np.array([
        [0.0, 0.0], [100.0, 0.0], [100.0, 100.0],
        [0.0, 100.0], [50.0, 30.0], [20.0, 80.0]
    ])
    H_true = np.array([
        [1.2, -0.1, 15.0],
        [0.05, 1.1, 25.0],
        [0.0001, 0.0002, 1.0]
    ], dtype=np.float64)
    dst = apply_homography(H_true, src)

    # --- [2. ACT] ---
    H_est = dlt_homography(src, dst)
    H_est_norm = normalize_homography(H_est)
    H_true_norm = normalize_homography(H_true)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(H_est_norm, H_true_norm, atol=1e-4, err_msg="DLT failed to recover ground truth H.")


def test_dlt_conditioning_collinear_degeneracy():
    """
    [INPUT]: 4 points placed on a straight line (collinear degenerate configuration).
    [OUTPUT]: DLT conditioning rank ratio s_8 / s_1 must be near zero (< 1e-4).
    """
    # --- [1. INPUT] ---
    # Points along line y = 2x
    src_collinear = np.array([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0], [40.0, 80.0]])
    dst_collinear = np.array([[15.0, 25.0], [25.0, 45.0], [35.0, 65.0], [45.0, 85.0]])

    # --- [2. ACT] ---
    rank_ratio = dlt_conditioning(src_collinear, dst_collinear)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert rank_ratio < 1e-3, f"Collinear points should have degenerate conditioning, got {rank_ratio}"


def test_reprojection_and_symmetric_error():
    """
    [INPUT]: Exact homography correspondence with one perturbed point (offset by 3px).
    [OUTPUT]:
        - reprojection_error on unperturbed points == 0.
        - reprojection_error on perturbed point == 3.0.
        - symmetric_transfer_error correctly combines forward and backward error.
    """
    # --- [1. INPUT] ---
    src = np.array([[0.0, 0.0], [10.0, 10.0]])
    H = np.eye(3, dtype=np.float64)
    dst = src.copy()
    dst[1] += [3.0, 0.0]  # 3 pixel offset along X

    # --- [2. ACT] ---
    fwd_err = reprojection_error(H, src, dst)
    sym_err = symmetric_transfer_error(H, src, dst)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert np.isclose(fwd_err[0], 0.0)
    assert np.isclose(fwd_err[1], 3.0)
    assert sym_err[1] > fwd_err[1], "Symmetric error must be >= forward error"


def test_footprint_ok_valid_and_invalid():
    """
    [INPUT]:
        - Valid affine homography.
        - Degenerate homography that collapses or explodes image area > max_area_ratio.
    [OUTPUT]:
        - footprint_ok returns True for normal transformation.
        - footprint_ok returns False for degenerate/explosive transformation.
    """
    # --- [1. INPUT] ---
    shape = (100, 100)
    H_valid = np.array([[1.1, 0.0, 10.0], [0.0, 1.1, 10.0], [0.0, 0.0, 1.0]])
    H_explosive = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]])  # 2500x area
    H_inverted = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])    # negative determinant

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    assert footprint_ok(H_valid, shape, max_area_ratio=25.0) is True
    assert footprint_ok(H_explosive, shape, max_area_ratio=25.0) is False
    assert footprint_ok(H_inverted, shape, max_area_ratio=25.0) is False


def test_ransac_homography_insufficient_points():
    """
    [INPUT]: Fewer than 4 point correspondences (e.g. 3 points).
    [OUTPUT]: Returns H=None, all False inliers, and iterations=0 without raising unhandled exceptions.
    """
    # --- [1. INPUT] ---
    src = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    dst = src.copy()

    # --- [2. ACT] ---
    H, inliers, info = ransac_homography(src, dst)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert H is None, "RANSAC must return None when N < 4"
    assert not inliers.any(), "Inlier mask must be all False"
    assert info["iterations"] == 0


def test_ransac_homography_with_outliers():
    """
    [INPUT]:
        - 20 points: 15 true inliers under translation (10, 5) and 5 random outliers.
    [OUTPUT]:
        - RANSAC must identify all 15 inliers and reject the 5 outliers.
        - Estimated homography matches ground-truth translation.
    """
    # --- [1. INPUT] ---
    np.random.seed(42)
    src_inliers = np.random.uniform(10, 90, size=(15, 2))
    dst_inliers = src_inliers + np.array([10.0, 5.0])

    src_outliers = np.random.uniform(10, 90, size=(5, 2))
    dst_outliers = np.random.uniform(200, 300, size=(5, 2))  # Wildly distant

    src_all = np.vstack([src_inliers, src_outliers])
    dst_all = np.vstack([dst_inliers, dst_outliers])

    # --- [2. ACT] ---
    H, inliers, info = ransac_homography(src_all, dst_all, thresh=2.0, seed=123)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert H is not None
    assert inliers[:15].sum() == 15, "All 15 true inliers should be accepted"
    assert inliers[15:].sum() == 0, "All 5 outliers must be rejected"
    assert np.isclose(info["rmse"], 0.0, atol=1e-3)


def test_refine_homography_improves_or_maintains_error():
    """
    [INPUT]: 10 point correspondences with slight random noise and initial DLT estimate.
    [OUTPUT]: Levenberg-Marquardt refinement must converge and achieve low reprojection RMSE.
    """
    # --- [1. INPUT] ---
    np.random.seed(99)
    src = np.random.uniform(10, 100, size=(10, 2))
    H_gt = np.array([[1.05, -0.02, 5.0], [0.03, 0.98, -8.0], [0.0001, 0.0001, 1.0]])
    dst = apply_homography(H_gt, src) + np.random.normal(0, 0.2, size=src.shape)
    H_init = dlt_homography(src, dst)

    # --- [2. ACT] ---
    H_refined = refine_homography(H_init, src, dst)

    # --- [3. OUTPUT ASSERTIONS] ---
    init_rmse = np.sqrt(np.mean(reprojection_error(H_init, src, dst) ** 2))
    refined_rmse = np.sqrt(np.mean(reprojection_error(H_refined, src, dst) ** 2))
    assert refined_rmse <= init_rmse + 1e-5, "LM refinement should not increase RMSE"


def test_required_iterations_extremes():
    """
    [INPUT]: Extreme inlier ratios (0.0 and 1.0).
    [OUTPUT]: 0.0 inlier ratio returns np.inf; 1.0 inlier ratio returns 0.0 iterations.
    """
    assert required_iterations(0.0, confidence=0.99) == np.inf
    assert required_iterations(1.0, confidence=0.99) == 0.0


def test_normalize_homography_zero_bottom_right():
    """
    [INPUT]: Homography matrix with H[2, 2] == 0.0.
    [OUTPUT]: normalize_homography falls back to normalizing by Frobenius norm.
    """
    # --- [1. INPUT] ---
    H = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 0.0]], dtype=np.float64)

    # --- [2. ACT] ---
    H_norm = normalize_homography(H)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_allclose(np.linalg.norm(H_norm), 1.0, atol=1e-7)


def test_dlt_insufficient_points_and_unnormalized():
    """
    [INPUT]: 3 point correspondences (< 4 required for DLT) and 4 points without normalization.
    [OUTPUT]: DLT returns None for 3 points; dlt_conditioning returns 0.0; unnormalized DLT recovers mapping.
    """
    # --- [1. INPUT] ---
    pts3 = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])

    # --- [2. ACT & 3. OUTPUT ASSERTIONS] ---
    assert dlt_homography(pts3, pts3) is None
    assert dlt_conditioning(pts3, pts3) == 0.0

    pts4 = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    H_unnorm = dlt_homography(pts4, pts4, normalize=False)
    np.testing.assert_allclose(H_unnorm, np.eye(3), atol=1e-5)


def test_refine_homography_zero_bottom_right():
    """
    [INPUT]: Normalized H with Hn[2, 2] == 0.0.
    [OUTPUT]: refine_homography immediately returns the input H unchanged.
    """
    # --- [1. INPUT] ---
    H_zero = np.zeros((3, 3), dtype=np.float64)
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    dst = src.copy()

    # --- [2. ACT] ---
    H_out = refine_homography(H_zero, src, dst)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_array_equal(H_out, H_zero)


def test_footprint_ok_opposite_homogeneous_signs():
    """
    [INPUT]: Homography mapping image corners to conflicting positive and negative w coordinates.
    [OUTPUT]: footprint_ok returns False.
    """
    # --- [1. INPUT] ---
    H_flip = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.01, -0.01, 0.0]], dtype=np.float64)

    # --- [2. ACT] ---
    is_ok = footprint_ok(H_flip, (100, 100))

    # --- [3. OUTPUT ASSERTIONS] ---
    assert is_ok is False


def test_dlt_homography_svd_linalg_error():
    """
    [INPUT]: 4 points, but SVD raises LinAlgError.
    [OUTPUT]: dlt_homography catches exception and returns None (lines 167-168).
    """
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    with patch("numpy.linalg.svd", side_effect=np.linalg.LinAlgError("SVD failed")):
        assert dlt_homography(pts, pts) is None


def test_refine_homography_solve_linalg_error():
    """
    [INPUT]: refine_homography where solve raises LinAlgError once.
    [OUTPUT]: Handles LinAlgError by scaling lambda (lines 310-312).
    """
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]])
    dst = src + 1.0
    orig_solve = np.linalg.solve
    call_count = 0
    def flaky_solve(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise np.linalg.LinAlgError("Singular")
        return orig_solve(*args, **kwargs)

    with patch("numpy.linalg.solve", side_effect=flaky_solve):
        H_ref = refine_homography(np.eye(3), src, dst, max_iters=2)
        assert H_ref.shape == (3, 3)


def test_refine_homography_max_iters_and_no_improvement():
    """
    [INPUT]: refine_homography with max_iters=1 where relative decrease >= 1e-10 is not met (branch 304->325)
             and when cost never improves (branch 307->322).
    """
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]])
    dst = src + 2.0
    # max_iters=1 finishes the outer loop naturally (304->325)
    H_ref1 = refine_homography(np.eye(3), src, dst, max_iters=1)
    assert H_ref1.shape == (3, 3)

    # Mock residual jacobian so that cost is always increasing
    with patch("panorama.homography._lm_residual_jacobian", return_value=(np.full(10, 100.0), np.eye(10, 8))):
        H_no_imp = refine_homography(np.eye(3), src, dst, max_iters=1)
        assert H_no_imp.shape == (3, 3)


def test_ransac_homography_dlt_sample_returns_none():
    """
    [INPUT]: dlt_homography returns None on initial sample.
    [OUTPUT]: RANSAC continues search (line 445) and successfully recovers homography on next trial.
    """
    src = np.array([[0.0, 0.0], [10.0, 0.0], [12.0, 10.0], [2.0, 11.0], [6.0, 5.0]])
    dst = src.copy()
    orig_dlt = dlt_homography
    calls = 0
    def dlt_flaky(s, d, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return orig_dlt(s, d, **kw)

    with patch("panorama.homography.dlt_homography", side_effect=dlt_flaky):
        H, inl, info = ransac_homography(src, dst, max_iters=20, seed=0)
        assert H is not None


def test_ransac_homography_best_cnt_less_than_four():
    """
    [INPUT]: 4 points, but threshold is 0.0 and dst has no match.
    [OUTPUT]: RANSAC finds 0 inliers (< 4) and returns (None, zeros_mask, info) (line 456).
    """
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    dst = np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0], [700.0, 800.0]])
    H, inl, info = ransac_homography(src, dst, thresh=1e-12, max_iters=2)
    assert H is None
    assert not inl.any()


def test_ransac_homography_refit_returns_none():
    """
    [INPUT]: dlt_homography returns None during consensus re-fit (len(s) > 4).
    [OUTPUT]: RANSAC breaks refit loop and returns None (lines 462, 468).
    """
    # --- [1. INPUT] ---
    t = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    src = np.stack([100.0 + 50.0 * np.cos(t), 100.0 + 50.0 * np.sin(t)], axis=1)
    dst = src.copy()
    orig_dlt = dlt_homography

    def dlt_fail_refit(s, d, **kw):
        if len(s) > 4:
            return None
        return orig_dlt(s, d, **kw)

    # --- [2. ACT] ---
    with patch("panorama.homography.dlt_homography", side_effect=dlt_fail_refit):
        H, inl, info = ransac_homography(src, dst, max_iters=5, seed=0)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert H is None
    assert not inl.any()


def test_ransac_homography_refit_loop_completes_all_iterations():
    """
    [INPUT]: 12 circular points where reprojection_error gradually expands inliers over 5 iterations.
    [OUTPUT]: Refit loop completes all 5 iterations without early break (branch 460->467).
    """
    # --- [1. INPUT] ---
    t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    src = np.stack([100.0 + 50.0 * np.cos(t), 100.0 + 50.0 * np.sin(t)], axis=1)
    dst = src.copy()
    orig_reproj = reprojection_error
    calls = 0

    def progressive_reproj(H, s, d):
        nonlocal calls
        calls += 1
        # Calls 2..6 correspond to refit loop iterations 0..4
        if 2 <= calls <= 6:
            n_inliers = 3 + calls  # 5, 6, 7, 8, 9
            err = np.full(len(s), 10.0)
            err[:n_inliers] = 1.0
            return err
        return orig_reproj(H, s, d)

    # --- [2. ACT] ---
    with patch("panorama.homography.reprojection_error", side_effect=progressive_reproj):
        H, inl, info = ransac_homography(src, dst, max_iters=1, seed=0)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert H is not None
    assert calls >= 7


def test_ransac_homography_refine_false():
    """
    [INPUT]: 5 matching points, ransac_homography called with refine=False.
    [OUTPUT]: Homography estimated without LM refinement (branch 471->473).
    """
    # --- [1. INPUT] ---
    t = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    src = np.stack([100.0 + 50.0 * np.cos(t), 100.0 + 50.0 * np.sin(t)], axis=1)
    dst = src.copy()

    # --- [2. ACT] ---
    H, inl, info = ransac_homography(src, dst, refine=False, seed=0)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert H is not None
    assert inl.all()
