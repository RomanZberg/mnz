# main.py

import argparse
from image_loader import load_and_preprocess_image
from color_quantizer import quantize_image
from facet_detector import detect_facets
from border_tracer import trace_and_simplify_borders
from label_placer import place_labels
from output_generator import generate_outputs


def main():
    parser = argparse.ArgumentParser(description="Generate a Paint-by-Numbers image from an input photo.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--colors", type=int, default=12, help="Number of colors for quantization.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs.")
    parser.add_argument("--resize_width", type=int, default=800, help="Resize image width for processing.")

    args = parser.parse_args()

    print("[1/6] Loading and preprocessing image...")
    image = load_and_preprocess_image(args.image_path, args.resize_width)

    print("[2/6] Quantizing colors...")
    quantized_image, palette = quantize_image(image, args.colors)

    print("[3/6] Detecting facets...")
    labeled_image, region_labels = detect_facets(quantized_image)

    print("[4/6] Tracing and simplifying borders...")
    borders = trace_and_simplify_borders(labeled_image)

    print("[5/6] Placing labels...")
    label_positions = place_labels(labeled_image)

    print("[6/6] Generating output...")
    generate_outputs(
        labeled_image,
        borders,
        label_positions,
        palette,
        output_dir=args.output_dir
    )

    print("✅ Paint-by-numbers generation complete! Check the output directory.")


if __name__ == "__main__":
    main()
