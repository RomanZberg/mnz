from abc import ABC, abstractmethod

import numpy as np


class SegmentationResult:
    def __init__(self, segment_labels, colors, quantized_image):
        self.segment_labels = segment_labels
        self.colors = colors
        self.quantized_image = quantized_image


class ImageSegmenter(ABC):

    @abstractmethod
    def segment(self, image_array: np.ndarray) -> SegmentationResult:
        raise NotImplementedError()
