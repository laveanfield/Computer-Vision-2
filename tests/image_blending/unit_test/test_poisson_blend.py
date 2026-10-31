"""
Unit tests for image_blending.poisson_blend module following the Input-Output testing model.
"""

import numpy as np
import pytest
from scipy.sparse import eye, csr_matrix
from image_blending.poisson_blend import (
    get_subimg,
    linear_solver,
    stitch_img,
    poisson_solver,
    poisson_blend,
)


def test_get_subimg():
    """
    [INPUT]: 5x5 sequential integer matrix, dims = (1, 4, 2, 5).
    [OUTPUT]: Cropped 3x3 matrix matching rows 1:4 and cols 2:5 of input.
    """
    # --- [1. INPUT] ---
    input_image = np.arange(25).reshape(5, 5)
    input_dims = (1, 4, 2, 5)
    expected_output = input_image[1:4, 2:5]

    # --- [2. ACT] ---
    actual_output = get_subimg(input_image, input_dims)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_output.shape == (3, 3), f"Expected shape (3, 3), got {actual_output.shape}"
    np.testing.assert_array_equal(actual_output, expected_output, err_msg="Incorrect sub-region extracted.")


def test_stitch_img():
    """
    [INPUT]:
        - Target: 6x6x3 zeros canvas.
        - Source: 2x2x3 patch filled with 5.0.
        - Dims: Insertion bounding box (2, 4, 1, 3).
    [OUTPUT]: Target modified such that region [2:4, 1:3] contains source patch; rest remains zeros.
    """
    # --- [1. INPUT] ---
    input_target = np.zeros((6, 6, 3), dtype=np.float32)
    input_source = np.full((2, 2, 3), 5.0, dtype=np.float32)
    input_dims = (2, 4, 1, 3)

    # --- [2. ACT] ---
    actual_output = stitch_img(input_source, input_target, input_dims)

    # --- [3. OUTPUT ASSERTIONS] ---
    np.testing.assert_array_equal(
        actual_output[2:4, 1:3, :], input_source, err_msg="Source was not pasted at correct target dims."
    )
    assert actual_output[0, 0, 0] == 0.0, "Background was unintentionally altered outside insertion region."


def test_linear_solver():
    """
    [INPUT]: 4x4 sparse Identity matrix A, vector b = [1, 2, 3, 4], target dims = (2, 2).
    [OUTPUT]: Solution x = A^-1 b = [1, 2, 3, 4] reshaped to 2D array [[1, 2], [3, 4]].
    """
    # --- [1. INPUT] ---
    sparse_A = eye(4, format="csr")
    vector_b = np.array([1.0, 2.0, 3.0, 4.0])
    target_dims = (2, 2)
    expected_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    # --- [2. ACT] ---
    actual_output = linear_solver(sparse_A, vector_b, target_dims)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert actual_output.shape == (2, 2), f"Expected reshaped output (2, 2), got {actual_output.shape}"
    np.testing.assert_allclose(actual_output, expected_matrix, err_msg="Linear solver failed identity verification.")


def test_poisson_solver_matrix_structure():
    """
    [INPUT]: 4x4 single channel problem with mask active at pixels (1, 1) and (1, 2).
    [OUTPUT]:
        - Sparse Laplacian matrix L of shape (16, 16) with diagonal = 4.0 inside mask and 1.0 outside.
        - Guidance field vector b of length 16.
    """
    # --- [1. INPUT] ---
    Hs, Ws = 4, 4
    input_source = np.full((Hs, Ws), 0.8, dtype=np.float64)
    input_target = np.zeros((Hs, Ws), dtype=np.float64)
    input_mask = np.zeros((Hs, Ws), dtype=np.float64)
    input_mask[1, 1:3] = 1.0  # 2 active masked pixels

    input_data = {
        "source": input_source,
        "mask": input_mask,
        "target": input_target,
        "dims": (0, Hs, 0, Ws),
    }

    # --- [2. ACT] ---
    L_out, b_out = poisson_solver(input_data, grad_mix=False)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert isinstance(L_out, csr_matrix), "Laplacian must be a CSR sparse matrix"
    assert L_out.shape == (16, 16), f"Expected shape (16, 16), got {L_out.shape}"
    assert len(b_out) == 16, f"Expected guidance length 16, got {len(b_out)}"

    diag_values = L_out.diagonal()
    mask_flat = input_mask.flatten(order="C")
    for idx, is_masked in enumerate(mask_flat > 0.99):
        expected_diag = 4.0 if is_masked else 1.0
        assert diag_values[idx] == expected_diag, f"Diagonal mismatch at index {idx}"


def test_poisson_solver_mixed_gradients():
    """
    [INPUT]: 4x4 problem where target has dominant edge gradient; grad_mix toggled True vs False.
    [OUTPUT]: Guidance vector b under grad_mix=True must differ from standard gradient guidance.
    """
    # --- [1. INPUT] ---
    Hs, Ws = 4, 4
    input_source = np.zeros((Hs, Ws), dtype=np.float64)
    input_target = np.zeros((Hs, Ws), dtype=np.float64)
    input_target[:, 2:] = 1.0  # Strong vertical edge in target
    input_mask = np.ones((Hs, Ws), dtype=np.float64)

    input_data = {
        "source": input_source,
        "mask": input_mask,
        "target": input_target,
        "dims": (0, Hs, 0, Ws),
    }

    # --- [2. ACT] ---
    _, b_standard = poisson_solver(input_data, grad_mix=False)
    _, b_mixed = poisson_solver(input_data, grad_mix=True)

    # --- [3. OUTPUT ASSERTIONS] ---
    assert not np.array_equal(b_standard, b_mixed), "Mixed gradient guidance should incorporate target edges."


def test_poisson_blend_execution():
    """
    [INPUT]: 3-channel image data (10x10 target, 4x4 source patch, dims = (3, 7, 3, 7)).
    [OUTPUT]:
        - Output shape matches target shape (10, 10, 3).
        - Pixel values bounded in [0.0, 1.0].
        - Preserves target pixels strictly outside the blended bounding box ROI.
    """
    # --- [1. INPUT] ---
    input_target = np.full((10, 10, 3), 0.3, dtype=np.float32)
    input_source = np.full((4, 4, 3), 0.7, dtype=np.float32)
    input_mask = np.zeros((4, 4, 3), dtype=np.float32)
    input_mask[1:3, 1:3, :] = 1.0
    input_dims = (3, 7, 3, 7)

    input_data = {
        "source": input_source,
        "mask": input_mask,
        "target": input_target,
        "dims": input_dims,
    }

    # --- [2. ACT] ---
    output_std = poisson_blend(input_data, grad_mix=False)
    output_mix = poisson_blend(input_data, grad_mix=True)

    # --- [3. OUTPUT ASSERTIONS] ---
    for res, mode in [(output_std, "Standard"), (output_mix, "Mixed")]:
        assert res.shape == input_target.shape, f"{mode} Poisson output shape mismatch"
        assert res.min() >= 0.0 and res.max() <= 1.0, f"{mode} Poisson output out of range [0, 1]"
        assert not np.isnan(res).any(), f"{mode} Poisson output contains NaNs"
        np.testing.assert_allclose(
            res[:3, :, :], input_target[:3, :, :], atol=1e-5,
            err_msg=f"{mode} Poisson altered background outside bounding box."
        )
