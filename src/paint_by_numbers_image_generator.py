import os
from typing import Tuple

import numpy as np
import skimage
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from skimage.measure import regionprops, label
from tqdm import tqdm

from src.sections import RegionDetector
from src.segmentation import ImageSegmenter
from src.utility import load_image, get_color_pallete


class PaintByNumbersImageGenerator:
    def __init__(self, image_segmenter: ImageSegmenter):
        self._image_segmenter = image_segmenter
        self._labels = []

    def generate_image(self, image_path: str, output_folder: str):
        image_array = load_image(image_path)

        os.makedirs(output_folder, exist_ok=True)

        image_array = load_image(image_path)

        segmentation_result = self._image_segmenter.segment(image_array)

        labeled_regions = segmentation_result.quantized_image
        labeled_regions = self._detect_regions_by_color(labeled_regions)
        labeled_regions = self._filter_small_regions(labeled_regions, min_size=350)  # tweak size

        color_to_number = self._assign_numbers(segmentation_result.colors)

        boundaries = skimage.segmentation.find_boundaries(
            labeled_regions, mode='outer'
        )

        boundaries = ~boundaries

        final_image = self._overlay_numbers(segmentation_result.quantized_image, labeled_regions, color_to_number)
        to_draw_image = Image.fromarray(boundaries)
        draw = ImageDraw.Draw(to_draw_image)
        font = ImageFont.load_default()

        for number, position in self._labels:
            draw.text(position, str(number), fill='black', font=font)

        # Save image
        output_image_path = os.path.join(output_folder, "paint_by_numbers_output.png")
        final_image.save(output_image_path)
        print(f"Paint-by-numbers image saved to {output_image_path}")

        to_draw_image.save(f'{output_folder}/paint.jpg')

        fig, ax =get_color_pallete(color_to_number)
        fig.savefig(f'{output_folder}/pallete.jpg')

        # Save legend
        self._save_legend(color_to_number, output_folder)

    def _detect_regions_by_color(self, quantized_image):
        h, w, _ = quantized_image.shape
        labeled_image = np.zeros((h, w), dtype=int)
        current_label = 1

        unique_colors = np.unique(quantized_image.reshape(-1, 3), axis=0)
        for color in unique_colors:
            mask = np.all(quantized_image == color, axis=-1)
            labeled_mask = label(mask, connectivity=1)
            labeled_mask[labeled_mask > 0] += current_label - 1
            labeled_image += labeled_mask
            current_label = labeled_image.max() + 1

        return labeled_image

    def _find_field_center(self, mask: np.ndarray, font_size) -> Tuple[int, int]:
        image_height = len(mask) - 1
        image_width = len(mask[0]) - 1

        padding = 5

        font_width, font_height = font_size

        font_width = font_width
        font_height = font_height

        def outside_image(index):
            index_y, index_x = index

            return (
                    (index_y < 0) or (index_y > image_height) or
                    (index_x < 0) or (index_x > image_width)
            )

        region_detector = RegionDetector()

        regions = region_detector.get_regions(mask)

        for region in regions:
            for middle_pixel in region.get_region_center_middle_positions():
                current_x, current_y = middle_pixel

                center_index = current_y, current_x

                high_x_index = current_y, (current_x + int(font_width / 2))
                low_x_index = current_y, (current_x - int(font_width / 2))
                high_y_index = (current_y + int(font_height / 2)), current_x
                low_y_index = (current_y - int(font_height / 2)), current_x

                if (
                        outside_image(center_index) or
                        (outside_image(high_x_index)) or
                        (outside_image(low_x_index)) or
                        (outside_image(high_y_index)) or
                        (outside_image(low_y_index))
                ):
                    continue

                if (
                        (mask[center_index] == True) and
                        (mask[high_x_index] == True) and
                        (mask[low_x_index] == True) and
                        (mask[high_y_index] == True) and
                        (mask[low_y_index] == True)
                ):
                    return current_x, current_y

        # print('no position found')
        # print('no center found')
        return 0, 0

    def _filter_small_regions(self, labeled_image, min_size=100):
        new_labeled = np.zeros_like(labeled_image)

        current_label = 1
        for region in regionprops(labeled_image):
            if region.area >= min_size:
                coords = region.coords
                for y, x in coords:
                    new_labeled[y, x] = current_label
                current_label += 1
        return new_labeled

    def _assign_numbers(self, cluster_centers):
        color_to_number = {tuple(center.astype(int)): idx + 1 for idx, center in enumerate(cluster_centers)}
        return color_to_number

    def _save_legend(self, color_to_number, output_folder):
        legend_path = os.path.join(output_folder, "legend.txt")
        with open(legend_path, "w") as f:
            for color, number in color_to_number.items():
                f.write(f"{number}: RGB{color}\n")
        print(f"Legend saved to {legend_path}")

    def _overlay_numbers(
            self,
            image_array: np.ndarray,
            labeled_regions: np.ndarray,
            color_to_number: dict,
            font_path: str = None,
            font_size: int = 20,
            num_samples: int = 200
    ) -> Image.Image:
        image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(image)
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        # Pro Region einmal durchgehen
        for region in tqdm(regionprops(labeled_regions)):
            # War bereits vorher so im code. Ich nehme an, dass ist der Hintergrund?!
            if region.label == 0:
                continue

            mask_full = (labeled_regions == region.label)

            color = tuple(image_array[mask_full][0])
            number = color_to_number.get(color, '?')

            font_size = font.getsize(str(number))

            # Hier wird meine neue Funktion aufgerufen.
            cx, cy = self._find_field_center(mask_full, font_size)

            # Übernahme von Pascal seinem Code

            font_width, font_height = font_size

            draw.text((cx - int(font_width / 2), cy - int(font_height / 2)), str(number), fill='black', font=font)
            self._labels.append((number, ((cx - int(font_width / 2), cy - int(font_height / 2)))))

        return image

    def _save_legend(self, color_to_number, output_folder):
        legend_path = os.path.join(output_folder, "legend.txt")
        with open(legend_path, "w") as f:
            for color, number in color_to_number.items():
                f.write(f"{number}: RGB{color}\n")
        print(f"Legend saved to {legend_path}")
