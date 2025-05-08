from src.paint_by_numbers_image_generator import PaintByNumbersImageGenerator
from src.segmentation.k_means import KMeansSegmenter

generator = PaintByNumbersImageGenerator(
    image_segmenter=KMeansSegmenter(16)
)

generator.generate_image(
    # image_path='./test-images-mnz/castle_and_guards.jpg',
    image_path='./test-images-mnz/beach_and_trees.jpg',
    output_folder='./target'
)
