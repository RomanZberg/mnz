import numpy as np
from sklearn.cluster import KMeans
import cv2

from src.segmentation import ImageSegmenter, SegmentationResult


class KMeansSegmenter(ImageSegmenter):

    def __init__(self, number_of_colors):
        self._number_of_colors: int = number_of_colors

    def segment(self, image_array: np.ndarray) -> SegmentationResult:
        # Apply Gaussian blur to smooth transitions and reduce small color noise
        blurred = cv2.GaussianBlur(image_array, (5, 5), 0)

        h, w, c = blurred.shape
        image_flat = blurred.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self._number_of_colors, random_state=42).fit(image_flat)
        labels = kmeans.labels_
        quantized_flat = kmeans.cluster_centers_[labels].astype('uint8')
        quantized_image = quantized_flat.reshape(h, w, 3)

        return SegmentationResult(
            segment_labels=labels.reshape(h, w),
            colors=kmeans.cluster_centers_,
            quantized_image=quantized_image
        )