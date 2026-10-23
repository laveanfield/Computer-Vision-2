"""
Image I/O and Interactive Preprocessing Module

This module provides utilities for loading, normalizing, displaying, and preprocessing image data for 
the image blending pipeline. It features an interactive graphical interface (GUI) using OpenCV that 
allows users to manually select the placement offset and adjust the scale of the source image over the 
target image.

Additionally, it handles boundary conditions by cropping or adjusting the source and mask patches if 
they extend beyond the target image boundaries.

"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_blending.config import SOURCE_DIR, MASK_DIR, TARGET_DIR

#=================================================
# Loading and showing image
#=================================================

def load_img(image_name: str, target_offset: list[int] | None = None, scale: float | None = None) -> dict:
    """
    Loads, normalizes, and optionally scales the source, mask, and target images.

    If `target_offset` is not provided, an interactive window will open to let 
    the user select the offset and scale manually.

    Args:
        image_name (str): The base name of the image file to load from the directories.
        target_offset (list[int] | None, optional): The (y, x) offset coordinates for 
            placing the source onto the target. Defaults to None.
        scale (float | None, optional): The scaling factor for the source and mask images. 
            Defaults to None.

    Returns:
        dict: A dictionary containing the normalized float32 image arrays and dimensions:
            - 'source' (np.ndarray): Normalized source image.
            - 'mask' (np.ndarray): Normalized mask image.
            - 'target' (np.ndarray): Normalized target image.
            - 'dims' (list[int] | tuple): The (y, x) offset coordinates.

    Raises:
        FileNotFoundError: If the source, mask, or target image cannot be found in 
            their respective directories.
    """
    image_data = {}
    source = cv2.imread(str(SOURCE_DIR / image_name))
    mask   = cv2.imread(str(MASK_DIR   / image_name))
    target = cv2.imread(str(TARGET_DIR / image_name))

    if source is None: raise FileNotFoundError(f"Source not found: {SOURCE_DIR / image_name}")
    if mask is None: raise FileNotFoundError(f"Mask not found: {MASK_DIR   / image_name}")
    if target is None: raise FileNotFoundError(f"Target not found: {TARGET_DIR / image_name}")

    # Normalize the images to [0, 1]
    image_data['source'] = cv2.normalize(source.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image_data['mask'] = cv2.normalize(mask.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image_data['target'] = cv2.normalize(target.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    if target_offset is None:
        dims, scale = select_offset(image_name)
        image_data['dims'] = dims
    else:
        image_data['dims'] = target_offset

    # Scale source and mask if user adjusted size
    if scale is not None and scale != 1.0:
            Hs, Ws = image_data['source'].shape[:2]
            new_H = max(1, int(round(Hs * scale)))
            new_W = max(1, int(round(Ws * scale)))
            image_data['source'] = cv2.resize(image_data['source'], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            image_data['mask']   = cv2.resize(image_data['mask'],   (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            print(f"Scaled source & mask by {scale:.2f}x → {new_H}x{new_W}")
    return image_data

def display_images(image_data: dict, title: str = "") -> None:
    """
    Displays the loaded images side-by-side using Matplotlib.

    Args:
        image_data (dict): A dictionary containing the image arrays 
            (e.g., 'source', 'mask', 'target').
        title (str, optional): The main title for the plot figure. Defaults to "".
    """
    keys = [k for k in image_data.keys() if k != "dims"]
    plt.figure(figsize=(6 * len(keys) , 6))
    if title:
        plt.suptitle(title)
    for i, key in enumerate(keys):
        img = image_data[key]
        plt.subplot(1, len(keys), i + 1)
        plt.title(key)
        plt.imshow(np.clip(img, 0, 1)[:, :, [2, 1, 0]]) # Convert BGR to RGB
        # plt.axis("off")
    plt.tight_layout()
    plt.show()

#=================================================
# Preprocessing
#=================================================

def preprocess_images(image_data: dict) -> dict:
    """
    Adjusts image coordinates and bounds to handle out-of-bounds placements.

    Corrects negative offsets by rolling and masking the out-of-bounds regions, 
    and crops the source and mask if they extend beyond the target dimensions.

    Args:
        image_data (dict): A dictionary containing:
            - 'source' (np.ndarray): The source image.
            - 'mask' (np.ndarray): The mask image.
            - 'target' (np.ndarray): The target image.
            - 'dims' (list[int]): The initial (y, x) or (Ho, Wo) offset.

    Returns:
        dict: A dictionary containing the adjusted 'source', 'mask', 'target' 
            images, and updated bounding 'dims' defined as (H_min, H_max, W_min, W_max).
    """
    source = image_data['source'].copy()
    mask = image_data['mask'].copy()
    target = image_data['target'].copy()
    
    # Get image shapes and offset dimensions
    Hs, Ws, _ = source.shape
    Ht, Wt, _ = target.shape
    Ho, Wo = image_data['dims']

    print(f"Source shape: {Hs}x{Ws}")
    print(f"Target shape: {Ht}x{Wt}")
    print(f"Offset (Ho, Wo): ({Ho}, {Wo})")
    print(f"Target region: H[{Ho}:{min(Ho+Hs,Ht)}] W[{Wo}:{min(Wo+Ws,Wt)}]")

    # Adjust the source and mask images based on the offsets (if negative)
    if Ho < 0:
        mask = np.roll(mask, shift=Ho, axis=0)
        source = np.roll(source, shift=Ho, axis=0)
        mask[Ho + Hs:,:,:] = 0
        source[Ho + Hs:,:,:] = 0
        Ho = 0
        
    
    if Wo < 0:
        mask = np.roll(mask, shift=Wo, axis=1)
        source = np.roll(source, shift=Wo, axis=1)
        mask[:, Wo + Ws:,:] = 0
        source[:, Wo + Ws:,:] = 0
        Wo = 0
    
    # Mask region on target
    H_min = Ho
    H_max = min(Ho + Hs, Ht)
    W_min = Wo
    W_max = min(Wo + Ws, Wt)

    # Crop source and mask if they exceed target dimensions
    source = source[0:min(Hs, Ht - Ho), 0:min(Ws, Wt - Wo), :]
    mask = mask[0:min(Hs, Ht - Ho), 0:min(Ws, Wt - Wo), :]

    return {'source': source, 'mask': mask, 'target': target, 'dims': (H_min, H_max, W_min, W_max)}

#=================================================
# Select offset
#=================================================

def select_offset(image_name: str) -> tuple[list[int], float]:
    """
    Opens an interactive GUI to select the offset and scale for blending.

    Controls:
        - Mouse move / Left-click : Choose placement position.
        - Scroll UP / press '+'   : Scale source and mask UP (+5% per step).
        - Scroll DOWN / press '-' : Scale source and mask DOWN (-5% per step).
        - Enter                   : Confirm selection and close window.

    Args:
        image_name (str): The base name of the image to process.

    Returns:
        tuple[list[int], float]: A tuple containing:
            - The selected offset as [y, x].
            - The final scale factor (float).
    """
    target_path = str(TARGET_DIR / image_name)
    source_path = str(SOURCE_DIR / image_name)
    mask_path   = str(MASK_DIR   / image_name)
    
    target = cv2.imread(target_path)
    source_orig = cv2.imread(source_path)
    mask_orig   = cv2.imread(mask_path)

    Ht, Wt = target.shape[:2]
    Hs_orig, Ws_orig = source_orig.shape[:2]

    state = {
        'offset': [0, 0],
        'scale': 1.0,
        'source': source_orig.copy(),
        'mask': mask_orig.copy(),
    }

    WIN = "Select Offset & Scale (scroll=resize, Enter=confirm)"

    def _redraw(y, x):
        src  = state['source']
        msk  = state['mask']
        Hs, Ws = src.shape[:2]
        display = target.copy()
        Hmax = min(y + Hs, Ht)
        Wmax = min(x + Ws, Wt)
        if Hmax > y and Wmax > x:
            src_crop  = src[0:Hmax - y, 0:Wmax - x]
            mask_crop = msk[0:Hmax - y, 0:Wmax - x] / 255.0
            roi = display[y:Hmax, x:Wmax]
            display[y:Hmax, x:Wmax] = (1 - mask_crop * 0.7) * roi + (mask_crop * 0.7) * src_crop
        cv2.rectangle(display, (x, y), (min(x+Ws,Wt), min(y+Hs,Ht)), (0, 255, 0), 2)
        sc = state['scale']
        cv2.putText(display,
                    f"Offset:({y},{x})  Scale:{sc:.2f}x ({int(Hs_orig*sc)}x{int(Ws_orig*sc)})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow(WIN, display)

    def _rescale(delta):
        state["scale"] = max(0.05, state["scale"] + delta)
        new_H = max(1, int(round(Hs_orig* state["scale"])))
        new_W = max(1, int(round(Ws_orig * state["scale"])))
        state["source"] = cv2.resize(source_orig, (new_W,  new_H), interpolation=cv2.INTER_LINEAR)
        state["mask"]   = cv2.resize(mask_orig,   (new_W,  new_H), interpolation=cv2.INTER_LINEAR)
        _redraw(*state["offset"])

    def mouse_callback(event, x, y, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            if event == cv2.EVENT_LBUTTONDOWN:
                state['offset'] = [y, x]
                # print(f"Selected offset: H0={y}, W0={x}")
            _redraw(state['offset'][0], state['offset'][1])

        elif event == cv2.EVENT_MOUSEWHEEL:
            _rescale(0.05 if flags > 0 else -0.05)

    cv2.imshow(WIN, target)
    cv2.setMouseCallback(WIN, mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:   # Enter to confirm
            break
        elif key == ord('+') or key == ord('='): 
            _rescale(+0.05)
        elif key == ord('-'): 
            _rescale(-0.05)

    cv2.destroyAllWindows()
    print(f"Confirmed — offset: {state['offset']}, scale: {state['scale']:.2f}x")
    return state['offset'], state['scale']