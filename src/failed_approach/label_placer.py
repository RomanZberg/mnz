# label_placer.py

import numpy as np
from skimage.measure import regionprops


def place_labels(labeled_image: np.ndarray):
    """
    Calculates the centroid position of each labeled region for placing a number label.

    Args:
        labeled_image (np.ndarray): A 2D array where each region has a unique integer label.

    Returns:
        Dict[int, Tuple[int, int]]: A dictionary mapping each region label to a (x, y) centroid position.
    """
    props = regionprops(labeled_image)
    label_positions = {}

    for region in props:
        label_id = region.label
        y, x = region.centroid  # row, col
        label_positions[label_id] = (int(x), int(y))

    return label_positions
