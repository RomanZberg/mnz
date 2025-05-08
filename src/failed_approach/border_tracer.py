# border_tracer.py

import numpy as np
from skimage import measure
from shapely.geometry import LineString


def trace_and_simplify_borders(labeled_image: np.ndarray, simplification_tolerance: float = 0.3):
    """
    Traces and simplifies the borders of each labeled region in the image.

    Args:
        labeled_image (np.ndarray): A 2D array where each region has a unique label.
        simplification_tolerance (float): Tolerance for polygon simplification. Higher means more simplification.

    Returns:
        Dict[int, List[Tuple[float, float]]]: Dictionary mapping each region label to its simplified border path.
    """
    borders = {}
    num_labels = labeled_image.max()

    for label_id in range(1, num_labels + 1):
        # Create a binary mask for the current region
        region_mask = labeled_image == label_id

        # Find contours (borders) of the region
        contours = measure.find_contours(region_mask.astype(np.uint8), level=0.5)

        if not contours:
            continue

        # Choose the longest contour (most representative)
        contour = max(contours, key=len)
        contour_line = LineString(contour[:, ::-1])  # Flip (row, col) to (x, y)

        # Simplify the contour
        simplified = contour_line.simplify(simplification_tolerance, preserve_topology=True)

        if simplified.is_empty or not simplified.is_ring:
            continue

        borders[label_id] = list(simplified.coords)

    return borders
