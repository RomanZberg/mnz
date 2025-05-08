from typing import List

import numpy as np
from skimage.measure import regionprops, label


class Region:
    def __init__(self, region_pixels=None):
        self._region_pixels: List[tuple[int, int]] = region_pixels

        if self._region_pixels is None:
            self._region_pixels = []

    def add_region_pixel(self, region_pixel: tuple[int, int]):
        self._region_pixels.append(region_pixel)

    def get_region_center_middle_positions(self):
        y_coords = list(set([pixel[1] for pixel in self._region_pixels]))

        y_groups = {
            y_cord: [] for y_cord in y_coords
        }

        for pixel in self._region_pixels:
            y_groups[pixel[1]].append(pixel[0])

        middle_positons = []
        for y_position, x_positions in y_groups.items():
            min_x = min(x_positions)
            max_x = max(x_positions)

            center_index = min_x + int((max_x - min_x) // 2)
            middle_positons.append((center_index, y_position))

        return middle_positons


class RegionDetector:
    def __init__(self):
        self._regions: List[Region] = []

        self._pixels_to_explore_for_region = []

        self._current_region = None
        self._in_region_lookup = None

        self._image_height = None
        self._image_width = None

        self._binary_mask = None

    def _coord_is_outside_image(self, coord):
        x, y = coord

        return (
                (x < 0) or (y < 0) or
                (x > self._image_width) or (y > self._image_height)
        )

    def _add_pixel_to_explore_for_pixel(self, pixel):
        x, y = pixel

        top = x, (y - 1)
        bottom = x, (y + 1)
        left = (x - 1), y
        right = (x + 1), y

        for neighbor_pixel in [top, bottom, left, right]:
            if (
                    (not self._coord_is_outside_image(neighbor_pixel)) and
                    self._binary_mask[neighbor_pixel[1], neighbor_pixel[0]] and
                    (not self._in_region_lookup[neighbor_pixel])
            ):
                self._pixels_to_explore_for_region.append(neighbor_pixel)

    def get_regions_old(self, binary_mask: np.ndarray):
        pass
        regions = []
        region_masks = regionprops(label(binary_mask, connectivity=1))

        for region_mask in region_masks:

            if region_mask.label == 0:
                continue

            region_mask = (region_mask.label == binary_mask)

            region_y_indexes, region_x_indexes = np.where(region_mask)

            mask_xy_coordinates = list(zip(region_x_indexes, region_y_indexes))
            regions.append(Region(mask_xy_coordinates))
        return regions

    def get_regions(self, binary_mask: np.ndarray) -> List[Region]:
        self._binary_mask = binary_mask
        region_y_indexes, region_x_indexes = np.where(binary_mask)

        self._image_width = binary_mask.shape[1] - 1
        self._image_height = binary_mask.shape[0] - 1

        mask_xy_coordinates = list(zip(region_x_indexes, region_y_indexes))

        self._in_region_lookup = {
            coords: False for coords in mask_xy_coordinates
        }

        regions = []

        while len(mask_xy_coordinates) > 0:
            region = Region()
            current_xy_coordinates = mask_xy_coordinates.pop()

            if self._in_region_lookup[current_xy_coordinates]:
                continue

            region.add_region_pixel(current_xy_coordinates)
            self._in_region_lookup[current_xy_coordinates] = True
            self._add_pixel_to_explore_for_pixel(
                current_xy_coordinates
            )

            while len(self._pixels_to_explore_for_region) > 0:
                current_xy_coordinates = self._pixels_to_explore_for_region.pop()
                current_x, current_y = current_xy_coordinates

                if (
                        self._in_region_lookup[current_xy_coordinates] or
                        (not binary_mask[current_y, current_x])
                ):
                    continue

                self._add_pixel_to_explore_for_pixel(
                    current_xy_coordinates
                )

                self._in_region_lookup[current_xy_coordinates] = True
                region.add_region_pixel(current_xy_coordinates)

            regions.append(region)
        return regions
