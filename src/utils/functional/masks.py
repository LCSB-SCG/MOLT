import numpy as np


def get_exclude_border_mask(
    shape: (int, int, int),
    top=True,
    bottom=True,
    left=True,
    right=True,
    front=True,
    back=True,
):
    """
    Create a mask that excludes the border of a 3D volume.

    Args:
        size (Tuple[int, int, int]): Size of the volume to create the mask for.
        top (bool): Whether to exclude the top border.
        bottom (bool): Whether to exclude the bottom border.
        left (bool): Whether to exclude the left border.
        right (bool): Whether to exclude the right border.
        front (bool): Whether to exclude the front border.
        back (bool): Whether to exclude the back border.

    Returns:
        np.ndarray: 3D mask with 1s in the center and 0s at the border.
    """
    mask = np.ones(shape)
    if len(shape) == 2:
        if top:
            mask[0, :] = 0
        if bottom:
            mask[-1, :] = 0
        if left:
            mask[:, 0] = 0
        if right:
            mask[:, -1] = 0
    elif len(shape) == 3:
        if top:
            mask[0, :, :] = 0
        if bottom:
            mask[-1, :, :] = 0
        if left:
            mask[:, 0, :] = 0
        if right:
            mask[:, -1, :] = 0
        if front:
            mask[:, :, 0] = 0
        if back:
            mask[:, :, -1] = 0

    return mask
