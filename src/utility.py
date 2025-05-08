import tqdm
from PIL import Image
import numpy as np
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
