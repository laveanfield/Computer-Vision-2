"""
Planar homography estimation, written from scratch (NumPy only).

    x' ~ H x ,   x = (x, y, 1)^T ,  H in R^{3x3} (defined up to scale -> 8 degrees of freedom)

"""
from __future__ import annotations

import numpy as np

#=================================================
# Basic geometry helpers
#=================================================

def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Projects 2-D points using a 3x3 homography matrix.

    Points mapped to infinity (homogeneous coordinate w ~ 0) result in NaN values.

    Args:
        H (np.ndarray): The 3x3 homography matrix.
        pts (np.ndarray): Array of shape (N, 2) containing the input [x, y] coordinates.

    Returns:
        np.ndarray: Array of shape (N, 2) containing the projected [x', y'] coordinates.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ np.asarray(H, dtype=np.float64).T
    w = ph[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ph[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out


def reprojection_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Computes the forward transfer error || pi(H x_i) - x'_i || in pixels.

    Args:
        H (np.ndarray): The 3x3 homography matrix.
        src (np.ndarray): Source coordinates of shape (N, 2).
        dst (np.ndarray): Destination coordinates of shape (N, 2).

    Returns:
        np.ndarray: Array of shape (N,) containing the error for each point. 
            Returns infinity if a point is sent to infinity.
    """
    err = np.linalg.norm(apply_homography(H, src) - np.asarray(dst, dtype=np.float64), axis=1)
    return np.where(np.isfinite(err), err, np.inf)


def symmetric_transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Computes the symmetric transfer error: d(x', Hx)^2 + d(x, H^-1 x')^2.

    Ref: Hartley & Zisserman, eq. 4.7.

    Args:
        H (np.ndarray): The 3x3 homography matrix mapping src to dst.
        src (np.ndarray): Source coordinates of shape (N, 2).
        dst (np.ndarray): Destination coordinates of shape (N, 2).

    Returns:
        np.ndarray: Array of shape (N,) containing the symmetric transfer error.
    """
    e1 = reprojection_error(H, src, dst)
    e2 = reprojection_error(np.linalg.inv(H), dst, src)
    return e1 ** 2 + e2 ** 2


def normalize_homography(H: np.ndarray) -> np.ndarray:
    """
    Normalizes the scale of the homography matrix.

    Ensures that H[2, 2] = 1. If H[2, 2] is near zero, falls back to normalizing 
    by the Frobenius norm.

    Args:
        H (np.ndarray): The 3x3 homography matrix.

    Returns:
        np.ndarray: The normalized 3x3 homography matrix.
    """
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        return H / H[2, 2]
    return H / np.linalg.norm(H)


#=================================================
# Direct Linear Transform
#=================================================

def normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies Hartley isotropic normalization to a set of points.

    Translates the centroid to the origin and scales the points so that the mean 
    distance to the origin is sqrt(2).

    Args:
        pts (np.ndarray): Input coordinates of shape (N, 2).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - normalized_pts (np.ndarray): The normalized coordinates.
            - T (np.ndarray): The 3x3 normalization transformation matrix where x_n = T x.
    """
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    T = np.array([[s, 0.0, -s * c[0]],
                  [0.0, s, -s * c[1]],
                  [0.0, 0.0, 1.0]])
    return (pts - c) * s, T


def dlt_homography(src: np.ndarray, dst: np.ndarray, normalize: bool = True) -> np.ndarray | None:
    """
    Computes the Direct Linear Transform (DLT) to estimate a homography.

    Each correspondence x -> x' provides two independent linear equations in the nine entries of H:
        [ 0  0  0 | -x -y -1 |  v x  v y  v ] h = 0
        [ x  y  1 |  0  0  0 | -u x -u y -u ] h = 0          with  x' = (u, v)
    Stacking N >= 4 correspondences forms A h = 0, where A is 2N x 9. The solution minimizing 
    ||A h|| subject to ||h|| = 1 is the right singular vector corresponding to the smallest 
    singular value of A.

    Args:
        src (np.ndarray): Source coordinates of shape (N, 2).
        dst (np.ndarray): Destination coordinates of shape (N, 2).
        normalize (bool, optional): If True, applies Hartley normalization before computing 
            the DLT. Defaults to True.

    Returns:
        np.ndarray | None: The estimated 3x3 homography matrix, or None if there are fewer 
            than 4 points or the SVD fails.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 4:
        return None

    if normalize:
        src_n, T1 = normalize_points(src)
        dst_n, T2 = normalize_points(dst)
    else:
        src_n, dst_n, T1, T2 = src, dst, np.eye(3), np.eye(3)

    x, y = src_n[:, 0], src_n[:, 1]
    u, v = dst_n[:, 0], dst_n[:, 1]
    zero, one = np.zeros(n), np.ones(n)
    A = np.zeros((2 * n, 9))
    A[0::2] = np.stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v], axis=1)
    A[1::2] = np.stack([x, y, one, zero, zero, zero, -u * x, -u * y, -u], axis=1)
    if A.shape[0] < 9:                       # exactly 4 points: pad so SVD returns all 9 right-singular vectors
        A = np.vstack([A, np.zeros((9 - A.shape[0], 9))])

    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    H = np.linalg.inv(T2) @ Vt[-1].reshape(3, 3) @ T1     # undo the normalisation
    return normalize_homography(H)


def dlt_conditioning(src: np.ndarray, dst: np.ndarray) -> float:
    """
    Evaluates the degeneracy of a correspondence set for DLT estimation.

    Calculates the ratio s_8 / s_1 of the second-smallest to the largest singular value 
    of the normalized DLT matrix A. The homography H is uniquely determined if and only if 
    rank(A) = 8, which means s_8 > 0. If points are collinear or collapse in either image, 
    A loses rank and the ratio approaches 0, indicating degeneracy.

    Args:
        src (np.ndarray): Source coordinates of shape (N, 2).
        dst (np.ndarray): Destination coordinates of shape (N, 2).

    Returns:
        float: The conditioning ratio s_8 / s_1. Returns 0.0 if there are fewer than 4 points.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4:
        return 0.0
    src_n, _ = normalize_points(src)
    dst_n, _ = normalize_points(dst)
    x, y, u, v = src_n[:, 0], src_n[:, 1], dst_n[:, 0], dst_n[:, 1]
    zero, one = np.zeros(len(x)), np.ones(len(x))
    A = np.zeros((2 * len(x), 9))
    A[0::2] = np.stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v], axis=1)
    A[1::2] = np.stack([x, y, one, zero, zero, zero, -u * x, -u * y, -u], axis=1)
    if A.shape[0] < 9:
        A = np.vstack([A, np.zeros((9 - A.shape[0], 9))])
    sv = np.linalg.svd(A, compute_uv=False)
    return float(sv[-2] / sv[0]) if sv[0] > 0 else 0.0


def footprint_ok(H: np.ndarray, shape: tuple[int, int], max_area_ratio: float = 25.0) -> bool:
    """
    Checks if the homography represents a plausible image-to-image mapping.

    Evaluates the transformation of the image rectangle through H to ensure:
      1. No point is sent to infinity (w must have the same sign at all corners), guaranteeing a convex quadrilateral.
      2. The quadrilateral preserves its orientation (positive signed area).
      3. The area scales by no more than `max_area_ratio`, preventing false matches that shrink to a speck or blow up.

    Args:
        H (np.ndarray): The 3x3 homography matrix.
        shape (tuple[int, int]): The (height, width) of the original image.
        max_area_ratio (float, optional): Maximum allowed ratio of mapped area to original area. Defaults to 25.0.

    Returns:
        bool: True if the mapping footprint is physically plausible, False otherwise.
    """
    H = np.asarray(H, dtype=np.float64)
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    ph = np.concatenate([corners, np.ones((4, 1))], axis=1) @ H.T
    wv = ph[:, 2]
    if not (np.all(wv > 1e-9) or np.all(wv < -1e-9)):
        return False
    q = ph[:, :2] / wv[:, None]
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))        # shoelace formula
    if not np.isfinite(area) or area <= 0:
        return False
    ratio = area / float((w - 1) * (h - 1))
    return 1.0 / max_area_ratio <= ratio <= max_area_ratio


#=================================================
# Levenberg-Marquardt refinement of the geometric error
#=================================================

def _lm_residual_jacobian(h8: np.ndarray, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the residuals and Jacobian for Levenberg-Marquardt refinement.

    Calculates the residuals r = pi(Hx) - x' and the Jacobian dr/dh, where 
    H = [[h0, h1, h2], [h3, h4, h5], [h6, h7, 1]].

    Args:
        h8 (np.ndarray): Array of the first 8 parameters of the normalized homography matrix.
        src (np.ndarray): Source coordinates of shape (N, 2).
        dst (np.ndarray): Destination coordinates of shape (N, 2).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - r (np.ndarray): Residual vector of shape (2N,).
            - J (np.ndarray): Jacobian matrix of shape (2N, 8).
    """
    H = np.append(h8, 1.0).reshape(3, 3)
    x, y = src[:, 0], src[:, 1]
    p = np.stack([x, y, np.ones_like(x)], axis=1) @ H.T
    w = p[:, 2]
    u, v = p[:, 0] / w, p[:, 1] / w
    n = len(src)

    r = np.empty(2 * n)
    r[0::2] = u - dst[:, 0]
    r[1::2] = v - dst[:, 1]

    J = np.zeros((2 * n, 8))
    iw = 1.0 / w
    J[0::2, 0], J[0::2, 1], J[0::2, 2] = x * iw, y * iw, iw
    J[0::2, 6], J[0::2, 7] = -u * x * iw, -u * y * iw
    J[1::2, 3], J[1::2, 4], J[1::2, 5] = x * iw, y * iw, iw
    J[1::2, 6], J[1::2, 7] = -v * x * iw, -v * y * iw
    return r, J


def refine_homography(H: np.ndarray, src: np.ndarray, dst: np.ndarray, max_iters: int = 30) -> np.ndarray:
    """
    Minimizes the geometric reprojection error using the Levenberg-Marquardt algorithm.

    Unlike the algebraic error used in DLT, this minimizes the true geometric distance
    sum_i ||pi(H x_i) - x'_i||^2. The optimization operates in Hartley-normalized coordinates.

    Args:
        H (np.ndarray): The initial estimate of the 3x3 homography matrix.
        src (np.ndarray): Source coordinates of shape (N, 2).
        dst (np.ndarray): Destination coordinates of shape (N, 2).
        max_iters (int, optional): Maximum number of LM iterations. Defaults to 30.

    Returns:
        np.ndarray: The refined 3x3 homography matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_n, T1 = normalize_points(src)
    dst_n, T2 = normalize_points(dst)
    Hn = T2 @ H @ np.linalg.inv(T1)
    if abs(Hn[2, 2]) < 1e-12:
        return H
    h = (Hn / Hn[2, 2]).ravel()[:8].copy()

    r, J = _lm_residual_jacobian(h, src_n, dst_n)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(max_iters):
        JTJ, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(12):
            try:
                delta = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            r_new, J_new = _lm_residual_jacobian(h + delta, src_n, dst_n)
            cost_new = float(r_new @ r_new)
            if np.isfinite(cost_new) and cost_new < cost:
                rel = (cost - cost_new) / max(cost, 1e-18)
                h, r, J, cost = h + delta, r_new, J_new, cost_new
                lam = max(lam / 10.0, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved or rel < 1e-10:
            break

    Hn_ref = np.append(h, 1.0).reshape(3, 3)
    return normalize_homography(np.linalg.inv(T2) @ Hn_ref @ T1)


#=================================================
# RANSAC
#=================================================

def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """
    Computes the signed double-area and 'sine' of the angle for triangle abc.

    Args:
        a (np.ndarray): Coordinate of point A.
        b (np.ndarray): Coordinate of point B.
        c (np.ndarray): Coordinate of point C.

    Returns:
        tuple[float, float]: A tuple containing the signed double-area (cross product) 
            and the sine of the angle (area / (|ab| |ac|)).
    """
    ab, ac = b - a, c - a
    cross = ab[0] * ac[1] - ab[1] * ac[0]
    denom = np.linalg.norm(ab) * np.linalg.norm(ac)
    return cross, (cross / denom if denom > 1e-12 else 0.0)


def _is_good_sample(s4: np.ndarray, d4: np.ndarray, min_sine: float = 0.02) -> bool:
    """
    Checks the geometric validity of a minimal sample of 4 point pairs.

    A sample is rejected if any 3 of the 4 points are nearly collinear in either image,
    or if the orientation of any defined triangle flips between views (indicating a reflection).

    Args:
        s4 (np.ndarray): Four source coordinates of shape (4, 2).
        d4 (np.ndarray): Four corresponding destination coordinates of shape (4, 2).
        min_sine (float, optional): The minimum sine value required to consider points non-collinear. Defaults to 0.02.

    Returns:
        bool: True if the sample is geometrically valid, False otherwise.
    """
    for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        cs, ss = _orientation(s4[i], s4[j], s4[k])
        cd, sd = _orientation(d4[i], d4[j], d4[k])
        if abs(ss) < min_sine or abs(sd) < min_sine or cs * cd < 0:
            return False
    return True


def required_iterations(inlier_ratio: float, confidence: float, sample_size: int = 4) -> float:
    """
    Computes the required number of RANSAC iterations to find an outlier-free sample.

    Solves for N in the inequality: 1 - (1 - w^s)^N >= p, 
    which simplifies to N >= log(1 - p) / log(1 - w^s), where w is the inlier ratio.

    Args:
        inlier_ratio (float): The estimated proportion of inliers.
        confidence (float): The desired probability of success (e.g., 0.995).
        sample_size (int, optional): The number of points in a minimal sample. Defaults to 4.

    Returns:
        float: The calculated minimum number of iterations needed.
    """
    ws = inlier_ratio ** sample_size
    if ws <= 1e-14:
        return np.inf
    if ws >= 1.0 - 1e-14:
        return 0.0
    return float(np.ceil(np.log(1.0 - confidence) / np.log(1.0 - ws)))


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    thresh: float = 3.0,
    confidence: float = 0.995,
    max_iters: int = 3000,
    seed: int | None = 0,
    refine: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """
    Robustly estimates a homography from putative correspondences using RANSAC.

    The algorithm repeatedly draws minimal 4-point samples, computes an exact DLT homography, 
    and counts inliers below a reprojection threshold. The number of trials adapts dynamically 
    based on the best observed inlier ratio. Finally, the model is re-fitted on all consensus 
    inliers using least squares and polished with Levenberg-Marquardt refinement.

    Args:
        src (np.ndarray): Source coordinates of shape (N, 2).
        dst (np.ndarray): Destination coordinates of shape (N, 2).
        thresh (float, optional): Inlier distance threshold in pixels. Defaults to 3.0.
        confidence (float, optional): Desired probability of drawing an outlier-free sample. Defaults to 0.995.
        max_iters (int, optional): Hard limit on the maximum number of iterations. Defaults to 3000.
        seed (int | None, optional): Random seed for reproducibility. Defaults to 0.
        refine (bool, optional): If True, applies LM non-linear refinement at the end. Defaults to True.

    Returns:
        tuple[np.ndarray | None, np.ndarray, dict]: A tuple containing:
            - H (np.ndarray | None): The estimated 3x3 homography matrix, or None if estimation failed.
            - inlier_mask (np.ndarray): A boolean array of length N indicating which correspondences are inliers.
            - info (dict): Diagnostic dictionary containing iteration counts, inlier statistics, and RMSE.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    info = {"iterations": 0, "n_matches": n, "n_inliers": 0, "inlier_ratio": 0.0,
            "rmse": float("nan"), "mean_error": float("nan")}
    if n < 4:
        return None, np.zeros(n, dtype=bool), info

    rng = np.random.default_rng(seed)
    best_inl, best_cnt, best_err_sum = np.zeros(n, dtype=bool), 0, np.inf
    needed, it = float(max_iters), 0

    while it < min(needed, max_iters):
        it += 1
        idx = rng.choice(n, size=4, replace=False)
        if not _is_good_sample(src[idx], dst[idx]):
            continue
        H = dlt_homography(src[idx], dst[idx])
        if H is None:
            continue
        err = reprojection_error(H, src, dst)
        inl = err < thresh
        cnt = int(inl.sum())
        err_sum = float(err[inl].sum())
        if cnt > best_cnt or (cnt == best_cnt and cnt > 0 and err_sum < best_err_sum):
            best_inl, best_cnt, best_err_sum = inl, cnt, err_sum
            needed = min(needed, required_iterations(cnt / n, confidence))
    info["iterations"] = it

    if best_cnt < 4:
        return None, np.zeros(n, dtype=bool), info

    # least-squares re-fit on the consensus set (iterate until the set stops growing)
    inl, H = best_inl, dlt_homography(src[best_inl], dst[best_inl])
    for _ in range(5):
        if H is None:
            break
        new_inl = reprojection_error(H, src, dst) < thresh
        if new_inl.sum() < 4 or np.array_equal(new_inl, inl):
            break
        inl, H = new_inl, dlt_homography(src[new_inl], dst[new_inl])
    if H is None:
        return None, np.zeros(n, dtype=bool), info

    # non-linear refinement of the geometric error, then final inlier recount
    if refine and inl.sum() >= 5:
        H = refine_homography(H, src[inl], dst[inl])
    err = reprojection_error(H, src, dst)
    inl = err < thresh

    info.update(n_inliers=int(inl.sum()), inlier_ratio=float(inl.mean()),
                rmse=float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("nan"),
                mean_error=float(err[inl].mean()) if inl.any() else float("nan"))
    return H, inl, info