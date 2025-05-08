import unittest
import matplotlib.pyplot as plt

import numpy as np

from src.sections import RegionDetector


class TestRegionDetector(unittest.TestCase):

    def test_get_regions(self):
        test_binary_mask = np.array([
            [False, True, True, True, True, False, False, False, False, False],
            [False, False, True, True, True, True, False, False, False, False],
            [False, True, True, True, True, False, False, False, False, False],
            [False, False, True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, True, True, True, False, False],
            [False, False, False, False, False, True, True, True, False, False],
            [False, False, False, False, False, True, True, True, False, False],
            [False, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False],
            [False, True, True, False, True, True, True, True, False, False],
            [False, True, True, False, False, True, True, True, False, False],
            [False, False, False, False, False, False, False, False, True, True],
            [False, False, False, False, False, False, False, False, True, True],
        ]).astype(int)

        existing_regions_in_mask = 5

        region_detector = RegionDetector()

        regions = region_detector.get_regions(test_binary_mask)

        test_binary_mask = test_binary_mask.astype(float)

        # Convert to RGB (shape: H x W x 3)
        rgb_mask = np.zeros((test_binary_mask.shape[0], test_binary_mask.shape[1], 3), dtype=np.uint8)

        # Set mask color (e.g. red for mask == 1)
        rgb_mask[test_binary_mask == 1] = [255, 0, 0]  # Red

        self.assertEqual(existing_regions_in_mask, len(regions))

        colors = [
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 255, 100],
            [0, 255, 150]
        ]

        for index, region in enumerate(regions):
            print(index)
            for pixel in region._region_pixels:
                rgb_mask[pixel[1], pixel[0]] = colors[index]

            for middle_pixel in region.get_region_center_middle_positions():
                rgb_mask[middle_pixel[1], middle_pixel[0]] = [255, 0, 0]

        plt.imshow(rgb_mask)
        plt.title('Binary Mask')
        plt.axis('off')  # Hide axes
        plt.show()

        pass


if __name__ == '__main__':
    unittest.main()
