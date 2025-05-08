import tqdm
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from PIL import ImageDraw, ImageFont
from skimage.measure import label, regionprops
import cv2
import os
from typing import Tuple
import random
from tqdm import tqdm

from src.sections import RegionDetector


def load_image(image_path) -> np.ndarray:
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def get_color_pallete(color_to_number_dict):
    fig, axs = plt.subplots(1, len(color_to_number_dict), figsize=(len(color_to_number_dict) * 3, 4))

    for ax, (rgb_255, label_number) in zip(axs, color_to_number_dict.items()):
        rgb_normalized = tuple(c / 255 for c in rgb_255)

        img = np.full((10, 10, 3), rgb_normalized)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{label_number}: RGB: {[int(value) for value in rgb_255]}")

    fig.tight_layout()

    return fig, axs
