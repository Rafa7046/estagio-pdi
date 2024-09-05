import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pickle

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the segmentation script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Segment images and save bounding boxes.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/images/train",
        help="Directory of images to segment."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/segmentation_results.pkl",
        help="Output file to save segmentation results."
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=50,
        help="Minimum area of connected components to consider in segmentation."
    )
    parser.add_argument(
        "--max_area",
        type=int,
        default=500,
        help="Maximum area of connected components to consider in segmentation."
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=8,
        help="Connectivity for connected component analysis."
    )
    return parser.parse_args()

def cca(image: np.ndarray, min_area: int, max_area: int, connectivity: int, min_aspect_ratio: float = 0.2, max_aspect_ratio: float = 1.5) -> List[Tuple[int, int, int, int]]:
    """
    Connected component analysis (CCA) with aspect ratio filtering for character segmentation.

    Args:
        image (np.ndarray): Input binary image.
        min_area (int): Minimum area of connected components to consider.
        max_area (int): Maximum area of connected components to consider.
        connectivity (int): Connectivity for CCA.
        min_aspect_ratio (float): Minimum aspect ratio to consider for characters.
        max_aspect_ratio (float): Maximum aspect ratio to consider for characters.

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes for valid characters.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=connectivity)
    
    bounding_boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h if h > 0 else 0

        # Filter by area and aspect ratio
        if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            bounding_boxes.append((x, y, x + w, y + h))

    return bounding_boxes

def projection_profiles(image: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Calculate horizontal and vertical projection profiles for character segmentation.

    Args:
        image (np.ndarray): Binarized image.

    Returns:
        Tuple[List[int], List[int]]: Horizontal and vertical projection profiles.
    """
    horizontal_projection = np.sum(image, axis=1)
    vertical_projection = np.sum(image, axis=0)
    
    return horizontal_projection, vertical_projection

def extract_line_boundaries(h_proj: List[int], threshold_ratio: float = 0.5) -> List[Tuple[int, int]]:
    """
    Extract the line boundaries from the horizontal projection profile using a dynamic threshold.
    
    Args:
        h_proj (List[int]): Horizontal projection profile.
        threshold_ratio (float): Ratio of the maximum value in the profile used as a threshold. Default is 0.2.
        
    Returns:
        List[Tuple[int, int]]: List of start and end positions for each line.
    """
    max_value = max(h_proj)
    threshold = max_value * threshold_ratio

    line_boundaries = []
    in_line = False
    
    for i, value in enumerate(h_proj):
        if value > threshold and not in_line:
            start = i
            in_line = True
        elif value <= threshold and in_line:
            end = i
            in_line = False
            line_boundaries.append((start, end))
    
    return line_boundaries

def segment_image(image: np.ndarray, min_area: int = 50, max_area: int = 500, connectivity: int = 8) -> List[Tuple[int, int, int, int]]:
    """
    Segment preprocessed image into characters using connected component analysis (CCA) and projection profiles.

    Args:
        preprocessed_image (np.ndarray): Preprocessed binary image (from preprocess_image).
        min_area (int): Minimum area of connected components to consider.
        max_area (int): Maximum area of connected components to consider.
        connectivity (int): Connectivity for CCA.
    
    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes for segmented characters.
    """
    # Calculate projection profiles
    h_proj, v_proj = projection_profiles(image)

    # Identify line boundaries using the horizontal projection profile
    line_boundaries = extract_line_boundaries(h_proj)

    # Segment characters within each line using vertical projection profiles and CCA
    bounding_boxes = []
    for (y1, y2) in line_boundaries:
        line_img = image[y1:y2, :]
        
        # Apply CCA to detect characters within the line
        char_bboxes = cca(line_img, min_area, max_area, connectivity)
        
        # Adjust bounding boxes to the original image coordinates
        adjusted_bboxes = [(x1, y1 + y_start, x2, y1 + y_end) for (x1, y_start, x2, y_end) in char_bboxes]
        bounding_boxes.extend(adjusted_bboxes)

    return bounding_boxes

def main():
    """
    Main function to perform image segmentation and save bounding boxes to a file.
    """
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    segmentation_results = {}

    # Perform segmentation for each image in the directory
    for image_path in input_dir.glob("*.png"):  # Assuming .png images; change if needed
        image = cv2.imread(str(image_path))
        bounding_boxes = segment_image(image, args.min_area, args.connectivity)
        segmentation_results[image_path.name] = bounding_boxes

    # Save segmentation results to a file
    with output_file.open('wb') as f:
        pickle.dump(segmentation_results, f)

    print(f"Segmentation results saved to {output_file}")

if __name__ == "__main__":
    main()
