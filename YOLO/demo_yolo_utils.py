"""
YOLO Utils Demo Script

This script demonstrates how to use the yolo_utils module to:
1. Load YOLO format bounding boxes
2. Convert them to pixel coordinates
3. Visualize them on an image

Usage:
    python demo_yolo_utils.py --label_file <path_to_label_file> --image_file <path_to_image_file>
"""

import os
import argparse
from yolo_utils import load_yolo_bboxes, visualize_bboxes


def parse_args():
    parser = argparse.ArgumentParser(description="Demo for YOLO bounding box utilities")
    
    parser.add_argument(
        "--label_file", 
        type=str, 
        default="train/labels/urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.txt",
        help="Path to the YOLO label file (.txt)"
    )
    
    parser.add_argument(
        "--image_file", 
        type=str, 
        default="train/images/urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg",
        help="Path to the corresponding image file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="visualization_output.jpg",
        help="Path to save the visualization output"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed information about the bounding boxes"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Ensure paths are correct
    if not os.path.exists(args.label_file):
        print(f"Error: Label file not found: {args.label_file}")
        return
        
    if not os.path.exists(args.image_file):
        print(f"Error: Image file not found: {args.image_file}")
        return
    
    # Load and convert bounding boxes
    try:
        bboxes = load_yolo_bboxes(args.label_file, args.image_file)
    except Exception as e:
        print(f"Error loading bounding boxes: {e}")
        return
    
    print(f"Successfully loaded {len(bboxes)} bounding boxes")
    
    # Print box information if verbose
    if args.verbose:
        for i, bbox in enumerate(bboxes):
            print(f"Box {i+1}:")
            print(f"  Class ID: {bbox['class_id']}")
            print(f"  Center: ({bbox['x_center']:.1f}, {bbox['y_center']:.1f}) pixels")
            print(f"  Size: {bbox['width']:.1f} x {bbox['height']:.1f} pixels")
            print(f"  Corners: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) to ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
    
    # Visualize the bounding boxes
    try:
        print(f"Visualizing bounding boxes and saving to {args.output}")
        visualize_bboxes(args.image_file, bboxes, args.output)
        print("Visualization complete")
    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    main()
