# facet_detector.py

import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects


def detect_facets(quantized_image: np.ndarray):
    """
    Detects and labels connected components (facets) in a quantized image.

    Args:
        quantized_image (np.ndarray): Quantized RGB image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A 2D labeled image where each facet has a unique label,
        and an array of region RGB colors indexed by label.
    """
    # Convert image to a single-channel ID map by hashing RGB values
    flat_image = quantized_image.reshape(-1, 3)
    unique_colors, color_indices = np.unique(flat_image, axis=0, return_inverse=True)
    indexed_image = color_indices.reshape(quantized_image.shape[:2])

    # Label connected regions of the same index
    labeled_image = label(indexed_image, connectivity=1)
    labeled_image = remove_small_objects(labeled_image, min_size=50)

    return labeled_image, unique_colors
