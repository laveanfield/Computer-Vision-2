"""
Shared test fixtures and helpers for image blending test suites.
Implements the Input-Output model for reusable test data setup.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest


@pytest.fixture
def synthetic_rgb_image() -> np.ndarray:
    """
    Input:
        A reproducible random 16x16x3 float32 array in [0.0, 1.0].
    Output:
        np.ndarray: Shape (16, 16, 3), dtype float32.
    """
    np.random.seed(42)
    return np.random.rand(16, 16, 3).astype(np.float32)


@pytest.fixture
def synthetic_blend_dict() -> dict:
    """
    Input:
        Pre-assembled synthetic image blending inputs:
        - target: 24x24x3 zero array (black canvas).
        - source: 8x8x3 ones array (white patch).
        - mask: 8x8x3 binary mask with center 4x4 active (1.0).
        - dims: Placement bounding box (4, 12, 4, 12).
    Output:
        dict: Containing 'source', 'mask', 'target', and 'dims'.
    """
    target = np.zeros((24, 24, 3), dtype=np.float32)
    source = np.ones((8, 8, 3), dtype=np.float32)

    mask = np.zeros((8, 8, 3), dtype=np.float32)
    mask[2:6, 2:6, :] = 1.0

    dims = (4, 12, 4, 12)
    return {
        "source": source,
        "mask": mask,
        "target": target,
        "dims": dims,
    }


@pytest.fixture
def mock_cv2_gui():
    """
    Input:
        Patches OpenCV window and mouse/keyboard interactions.
    Output:
        dict: Mock objects for 'imshow', 'setMouseCallback', 'waitKey', 'destroyAllWindows'.
    """
    with patch("cv2.imshow") as mock_imshow, \
         patch("cv2.setMouseCallback") as mock_mouse, \
         patch("cv2.waitKey", return_value=13) as mock_wait, \
         patch("cv2.destroyAllWindows") as mock_destroy:
        yield {
            "imshow": mock_imshow,
            "setMouseCallback": mock_mouse,
            "waitKey": mock_wait,
            "destroyAllWindows": mock_destroy,
        }


@pytest.fixture
def mock_plt_show():
    """
    Input:
        Patches matplotlib.pyplot.show.
    Output:
        MagicMock: Mock instance to verify display invocations without blocking.
    """
    with patch("matplotlib.pyplot.show") as mock_show:
        yield mock_show
