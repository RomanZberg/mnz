# color_quantizer.py

import numpy as np
from sklearn.cluster import KMeans

def quantize_image(image: np.ndarray, num_colors: int):
    """
    Applies K-means clustering to reduce the number of colors in the image.

    Args:
        image (np.ndarray): Input RGB image as a NumPy array.
        num_colors (int): Number of colors to reduce the image to.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int, int]]]: Quantized image (NumPy array),
        and list of RGB tuples representing the color palette.
    """
    h, w, _ = image.shape
    pixels = image.reshape((-1, 3))  # Flatten the image into a 2D array of RGB pixels

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixels)
    palette = kmeans.cluster_centers_.astype(np.uint8)

    # Map each pixel to its nearest palette color
    quantized_pixels = palette[labels].reshape((h, w, 3))

    print("Palette:", palette)

    return quantized_pixels, palette.tolist()
