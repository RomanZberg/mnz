# output_generator.py

import os
import svgwrite
import numpy as np
from PIL import Image


def generate_outputs(
    labeled_image: np.ndarray,
    borders: dict,
    label_positions: dict,
    palette: list,
    output_dir: str
):
    """
    Generates SVG output for the paint-by-numbers image, including borders and labels.

    Args:
        labeled_image (np.ndarray): 2D array with unique labels for each region.
        borders (dict): Dict mapping region labels to simplified border coordinates.
        label_positions (dict): Dict mapping region labels to label coordinates.
        palette (list): List of RGB tuples.
        output_dir (str): Directory where outputs will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    height, width = labeled_image.shape
    svg_path = os.path.join(output_dir, "paint_by_numbers.svg")

    dwg = svgwrite.Drawing(svg_path, size=(width, height))

    # Draw the regions
    for label_id, border in borders.items():
        color_index = (labeled_image == label_id).astype(int).sum()
        if not border:
            continue

        path_data = ["M {} {}".format(*border[0])]
        for point in border[1:]:
            path_data.append("L {} {}".format(*point))
        path_data.append("Z")

        dwg.add(dwg.path(
            d=" ".join(path_data),
            stroke="black",
            fill="white",
            stroke_width=0.5
        ))

    # Draw numbers
    for label_id, (x, y) in label_positions.items():
        dwg.add(dwg.text(
            str(label_id),
            insert=(x, y),
            fill='black',
            font_size='6px',
            text_anchor='middle',
            alignment_baseline='central'
        ))

    # Add color legend
    legend_x, legend_y = 10, height + 20
    for idx, color in enumerate(palette):
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        dwg.add(dwg.rect(insert=(legend_x + idx * 60, legend_y), size=(50, 15), fill=hex_color))
        dwg.add(dwg.text(
            str(idx + 1),
            insert=(legend_x + idx * 60 + 25, legend_y + 25),
            font_size='10px',
            fill='black',
            text_anchor='middle'
        ))

    dwg.save()
    print(f"✅ SVG file saved to {svg_path}")

    # Optional: Save a reference PNG for debugging
    ref_img = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(labeled_image)
    for i, label_id in enumerate(unique_labels[unique_labels > 0]):
        ref_img[labeled_image == label_id] = palette[i % len(palette)]
    Image.fromarray(ref_img).save(os.path.join(output_dir, "reference_colored.png"))
    print(f"✅ Reference image saved to {os.path.join(output_dir, 'reference_colored.png')}")
