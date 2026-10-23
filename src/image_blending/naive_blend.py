"""
Naive Image Blending Module

This module provides a straightforward implementation of naive alpha blending for image compositing. 
It uses a spatial mask to linearly interpolate between a source image patch and a specific region of 
a target image.


"""
import numpy as np

def naive_blend(image_data: dict) -> np.ndarray:
    """
    Blends a source image into a target image using naive alpha blending.

    Args:
        image_data (dict): A dictionary containing the required image data components:
            - 'source' (np.ndarray): The source image array to be blended.
            - 'mask' (np.ndarray): The blending mask array (values usually in [0, 1]).
            - 'target' (np.ndarray): The full target image array.
            - 'dims' (tuple): The placement bounding box in the target image defined 
              as (H_min, H_max, W_min, W_max).

    Returns:
        np.ndarray: The resulting target image with the source blended onto it.
    """
    source = image_data['source']
    mask = image_data['mask']
    target = image_data['target'].copy()
    dims = image_data['dims']

    target[dims[0]:dims[1], dims[2]:dims[3], :] = target[dims[0]:dims[1],dims[2]:dims[3],:] * (1 - mask) + source * mask

    return target