import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage import filters
from tqdm import tqdm

from utils import TQDM_FORMAT

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the preprocessing script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess images for OCR.")
    parser.add_argument("--data_dir", type=str, default="data/images/train", help="Directory of images to preprocess.")
    parser.add_argument("--output_dir", type=str, default="outputs/preprocessed", help="Output dir to save preprocessed images.")
    parser.add_argument("--blur_kernel", type=int, default=7, help="Kernel size for Gaussian blur.")
    parser.add_argument("--threshold_method", type=str, default="otsu", choices=['otsu', 'adaptive'], help="Thresholding method.")
    parser.add_argument("--morph_kernel", type=int, default=3, help="Kernel size for morphological operations.")
    return parser.parse_args()

def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from a file path and convert it to grayscale.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        np.ndarray: Grayscale image.
    """
    image = cv2.imread(str(image_path))
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the image contrast.

    Args:
        image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Enhanced image with improved contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image


def reduce_noise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise in the image.

    Args:
        image (np.ndarray): Grayscale image.
        kernel_size (int): Size of the Gaussian kernel. Default is 5.

    Returns:
        np.ndarray: Blurred image.
    """
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def apply_threshold(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """
    Apply thresholding and detect if the image should be inverted (white text on black background).

    Args:
        image (np.ndarray): Grayscale image.
        method (str): Thresholding method, either 'otsu' or 'adaptive'. Default is 'otsu'.

    Returns:
        np.ndarray: Binarized and possibly inverted image.
    """
    invert = cv2.mean(image)[0] > 127
    
    # Apply thresholding
    if method == 'otsu':
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
    
    if invert:
        binary_image = cv2.bitwise_not(binary_image)
    
    return binary_image

def morphological_operations(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply advanced morphological operations and projection profile analysis to improve character segmentation.

    Args:
        image (np.ndarray): Binarized image.
        kernel_size (int): Size of the kernel for morphological operations.

    Returns:
        np.ndarray: Image after advanced segmentation.
    """
    # Closing operation to join broken parts of characters
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return closed_image

def preprocess_image(image_path: Path, blur_kernel: int = 5, threshold_method: str = 'otsu', morph_kernel: int = 3) -> np.ndarray:
    """
    Complete preprocessing pipeline for OCR, including noise reduction, thresholding, and morphological operations.

    Args:
        image_path (Path): Path to the image file.
        blur_kernel (int): Kernel size for Gaussian blur. Default is 5.
        threshold_method (str): Thresholding method ('otsu' or 'adaptive'). Default is 'otsu'.
        morph_kernel (int): Kernel size for morphological operations. Default is 3.

    Returns:
        np.ndarray: Preprocessed image ready for OCR.
    """
    image = load_image(image_path)
    image = enhance_image(image)
    image = reduce_noise(image, kernel_size=blur_kernel)
    image = apply_threshold(image, method=threshold_method)
    image = morphological_operations(image, kernel_size=morph_kernel)
    return image

def main(args: argparse.Namespace) -> None:
    """
    Main function to preprocess images and save the results to a pickle file.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Convert the data directory and output file path to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate through each image in the specified directory and preprocess it
    for image_path in tqdm(data_dir.iterdir(), bar_format=TQDM_FORMAT, desc="Preprocessing images"):
        if image_path.is_file():
            processed_image = preprocess_image(
                image_path,
                blur_kernel=args.blur_kernel,
                threshold_method=args.threshold_method,
                morph_kernel=args.morph_kernel
            )
            # Write image to output directory
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), processed_image)

    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
