"""
Poisson Image Editing Module

This module provides a NumPy and SciPy-based implementation of Poisson Image Editing for seamless 
image blending. It solves the Poisson equation using sparse linear systems to blend a source image 
patch into a target image without visible seams.

The algorithm enforces the gradient field of the source image while keeping the boundary pixels equal 
to the target image (Dirichlet boundary conditions). It also supports "mixed gradient" blending, which 
preserves the dominant texture of both the source and the target images by selecting the stronger gradient 
at each pixel.

"""
import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse import diags, csr_matrix

#=================================================
# Helper functions
#=================================================

def get_subimg(image: np.ndarray, dims: tuple) -> np.ndarray:
    """
    Extracts a sub-region from an image based on provided dimensions.

    Args:
        image (np.ndarray): The input image from which to extract the region.
        dims (tuple): A tuple of four integers (H_min, H_max, W_min, W_max) 
            representing the bounding box of the region to extract.

    Returns:
        np.ndarray: The cropped sub-image.
    """
    return image[dims[0]:dims[1], dims[2]:dims[3]]

def linear_solver(A: csr_matrix, b: np.ndarray, dims: tuple) -> np.ndarray:
    """
    Solves a sparse linear system Ax = b and reshapes the result.

    Args:
        A (csr_matrix): The sparse matrix representing the linear system (e.g., Laplacian).
        b (np.ndarray): The right-hand side vector (e.g., guidance field).
        dims (tuple): A tuple representing the target dimensions (height, width) 
            to reshape the 1D output vector into a 2D image array.

    Returns:
        np.ndarray: The 2D array containing the reshaped solution of the linear system.
    """
    x = linalg.spsolve(A.tocsc(), b)
    return np.reshape(x, (dims[0], dims[1]))

def stitch_img(source: np.ndarray, target: np.ndarray, dims: tuple) -> np.ndarray:
    """
    Pastes a source image region into a specific location of a target image.

    Args:
        source (np.ndarray): The source image patch to be pasted.
        target (np.ndarray): The target image where the source will be inserted.
        dims (tuple): A tuple of four integers (H_min, H_max, W_min, W_max) 
            indicating the placement region in the target image.

    Returns:
        np.ndarray: The modified target image with the source pasted into it.
    """
    target[dims[0]:dims[1], dims[2]:dims[3], :] = source
    return target

#=================================================
# Build sparse Laplacian matrix and guidance vector
#=================================================

def poisson_solver(image_data: dict, grad_mix: bool) -> list:
    """
    Constructs the sparse Laplacian matrix and guidance vector for Poisson blending.

    Finds L (Laplacian matrix) and b (guidance field) of the equation: L.f = b.

    Args:
        image_data (dict): A dictionary containing single-channel data:
            - 'source' (np.ndarray): The source image channel.
            - 'mask' (np.ndarray): The blending mask channel (values near 1 indicate blending).
            - 'target' (np.ndarray): The full target image channel.
            - 'dims' (tuple): The bounding box dimensions (H_min, H_max, W_min, W_max).
        grad_mix (bool): If True, uses mixed gradients (keeps the stronger gradient 
            between the source and target). If False, uses only the source gradients.

    Returns:
        list: A list containing two elements:
            - [0] (csr_matrix): The sparse Laplacian block matrix (L).
            - [1] (np.ndarray): The flattened guidance field vector (b).
    """
    mask = image_data['mask']
    Hs, Ws = mask.shape
    num_pixels = Hs * Ws

    source = image_data['source'].flatten(order='C')
    target_subimg = get_subimg(image_data['target'], image_data['dims']).flatten(order='C')
    mask_flat = mask.flatten(order='C')
    mask_bool = mask_flat > 0.99
    # Build the sparse Laplacian block matrix 
    diag_main = np.where(mask_bool, 4.0, 1.0) # 4 in masked regions, 1 in unmasked regions

    diag_up = np.zeros(num_pixels - Ws)
    diag_down = np.zeros(num_pixels - Ws)
    diag_left = np.zeros(num_pixels - 1)
    diag_right = np.zeros(num_pixels - 1)

    ## Neighbor up
    valid_up = mask_bool[Ws:]
    diag_up = np.where(valid_up, -1.0, 0.0)

    ## Neighbor down
    valid_down = mask_bool[:num_pixels - Ws]
    diag_down = np.where(valid_down, -1.0, 0.0)

    ## Neighbor left
    valid_left = mask_bool[1:].copy()
    valid_left[Ws-1::Ws] = False  # Invalidate left neighbors at the start of each row
    diag_left = np.where(valid_left, -1.0, 0.0)

    ## Neighbor right
    valid_right = mask_bool[:num_pixels - 1].copy()
    valid_right[Ws-1::Ws] = False  # Invalidate right neighbors at the start of each row
    diag_right = np.where(valid_right, -1.0, 0.0)


    laplacian = diags(
        [diag_main, diag_up, diag_down, diag_left, diag_right], 
        [0, -Ws, Ws, -1, 1], 
        shape=(num_pixels, num_pixels), format='csr'
    )
    
    # Build guidance field
    guidance_field = target_subimg.copy()

    src_2d = source.reshape(Hs, Ws)
    tgt_2d = target_subimg.reshape(Hs, Ws)

    ## Gradient source
    grad_up_s = src_2d - np.roll(src_2d, shift=1, axis=0); grad_up_s[0, :] = src_2d[0, :]
    grad_down_s = src_2d - np.roll(src_2d, shift=-1, axis=0); grad_down_s[-1, :] = src_2d[-1, :]
    grad_left_s = src_2d - np.roll(src_2d, shift=1, axis=1); grad_left_s[:, 0] = src_2d[:, 0]
    grad_right_s =src_2d - np.roll(src_2d, shift=-1, axis=1); grad_right_s[:, -1] = src_2d[:, -1]
    
    if grad_mix:
        grad_up_t = tgt_2d - np.roll(tgt_2d, shift=1, axis=0); grad_up_t[0, :] = tgt_2d[0, :]
        grad_down_t = tgt_2d - np.roll(tgt_2d, shift=-1, axis=0); grad_down_t[-1, :] = tgt_2d[-1, :]
        grad_left_t = tgt_2d - np.roll(tgt_2d, shift=1, axis=1); grad_left_t[:, 0] = tgt_2d[:, 0]
        grad_right_t = tgt_2d - np.roll(tgt_2d, shift=-1, axis=1); grad_right_t[:, -1] = tgt_2d[:, -1]

        _compare = lambda x, y: np.where(np.abs(x) > np.abs(y), x, y)
        g_up = _compare(grad_up_s, grad_up_t)
        g_down = _compare(grad_down_s, grad_down_t)
        g_left = _compare(grad_left_s, grad_left_t)
        g_right = _compare(grad_right_s, grad_right_t)
    else:
        g_up = grad_up_s
        g_down = grad_down_s
        g_left = grad_left_s
        g_right = grad_right_s

    div = (g_up + g_down + g_left + g_right).flatten(order='C')
    guidance_field = np.where(mask_bool, div, target_subimg)

    return [laplacian, guidance_field]

#=================================================
# Poisson blending
#=================================================

def poisson_blend(image_data: dict, grad_mix: bool = False) -> np.ndarray:
    """
    Seamlessly blends a source image into a target image using Poisson Image Editing.

    Constructs and solves the Poisson equation for each color channel independently 
    to enforce smooth transitions matching the gradient fields.

    Args:
        image_data (dict): A dictionary containing the multi-channel image data:
            - 'source' (np.ndarray): The multi-channel source image.
            - 'mask' (np.ndarray): The multi-channel blending mask.
            - 'target' (np.ndarray): The full multi-channel target image.
            - 'dims' (tuple): The placement bounding box (H_min, H_max, W_min, W_max).
        grad_mix (bool, optional): If True, applies mixed gradient blending. 
            Defaults to False.

    Returns:
        np.ndarray: The final blended image, bounded within the range [0, 1].
    """
    equation_param = []
    ch_data = {}
    # Construct Poisson equation for each channel
    for ch in range(3):
        ch_data['source'] = image_data['source'][:, :, ch]
        ch_data['mask'] = image_data['mask'][:, :, ch]
        ch_data['target'] = image_data['target'][:, :, ch]
        ch_data['dims'] = image_data['dims']
        equation_param.append(poisson_solver(ch_data, grad_mix))

    # Solve the Poisson equation for each channel
    image_result = np.empty_like(image_data['source'])
    for i in range(3):
        image_result[:, :, i] = linear_solver(equation_param[i][0], equation_param[i][1], (image_data['source'].shape[0], image_data['source'].shape[1]))
        
    image_result = stitch_img(image_result, image_data['target'].copy(), image_data['dims'])

    return np.clip(image_result, 0, 1)