# image_loader.py

from PIL import Image
import numpy as np
import os
from scipy.ndimage import gaussian_filter

def load_and_preprocess_image(image_path: str, resize_width: int = 800) -> np.ndarray:
    """
    Loads an image from the given path, resizes it while maintaining aspect ratio,
    and converts it to a NumPy array.

    Args:
        image_path (str): Path to the input image file.
        resize_width (int): Desired width to resize the image for faster processing.

    Returns:
        np.ndarray: Resized image as a NumPy array in RGB format.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    # image = gaussian_filter(image, sigma=1)  # Smooths transitions
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(resize_width * aspect_ratio)

    image = image.resize((resize_width, new_height), Image.LANCZOS)
    image_np = np.array(image)

    return image_np
