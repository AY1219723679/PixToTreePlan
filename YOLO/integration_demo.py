"""
YOLO Trunk Detection Integration Demo

This script demonstrates how to integrate tree trunk detection using YOLO labels
with the PixToTreePlan pipeline.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the detect_tree_trunks package
from main.detect_tree_trunks.trunk_detection import detect_tree_trunks, enhance_ground_mask_with_trunk_detections, create_trunk_mask

# Import from YOLO utils
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'YOLO'))
from yolo_utils import visualize_bboxes


def parse_args():
    parser = argparse.ArgumentParser(description="Demo for tree trunk detection integration")
    
    parser.add_argument(
        "--image_path", 
        type=str, 
        default=None,
        help="Path to the image file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="trunk_detection_output",
        help="Directory to save the output files"
    )
    
    return parser.parse_args()


def create_visualization_grid(original_img, trunk_detections, ground_mask, enhanced_mask):
    """
    Create a visualization grid showing the detection and mask enhancement process.
    
    Args:
        original_img (PIL.Image): Original image
        trunk_detections (list): Detected trunk bounding boxes
        ground_mask (numpy.ndarray): Original ground mask
        enhanced_mask (numpy.ndarray): Enhanced ground mask with trunks
        
    Returns:
        numpy.ndarray: Visualization grid as RGB image
    """
    # Convert PIL image to numpy array if needed
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img)
    
    # Create copy of original image for drawing boxes
    img_with_boxes = original_img.copy()
    
    # Draw bounding boxes
    for trunk in trunk_detections:
        x1, y1 = int(trunk['x1']), int(trunk['y1'])
        x2, y2 = int(trunk['x2']), int(trunk['y2'])
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Create colored masks for visualization
    ground_mask_vis = np.zeros((*ground_mask.shape, 3), dtype=np.uint8)
    ground_mask_vis[ground_mask > 0] = [0, 0, 255]  # Blue for ground
    
    enhanced_mask_vis = np.zeros((*enhanced_mask.shape, 3), dtype=np.uint8)
    enhanced_mask_vis[enhanced_mask > 0] = [0, 255, 0]  # Green for enhanced mask
    
    # Add semi-transparency to the masks
    alpha = 0.5
    ground_overlay = cv2.addWeighted(original_img, 1 - alpha, ground_mask_vis, alpha, 0)
    enhanced_overlay = cv2.addWeighted(original_img, 1 - alpha, enhanced_mask_vis, alpha, 0)
    
    # Create a 2x2 grid for visualization
    h, w = original_img.shape[:2]
    grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    
    # Place images in grid
    grid[:h, :w] = original_img
    grid[:h, w:w*2] = img_with_boxes
    grid[h:h*2, :w] = ground_overlay
    grid[h:h*2, w:w*2] = enhanced_overlay
    
    return grid


def main():
    args = parse_args()
    
    # If no image path provided, use a default from the YOLO train directory
    if args.image_path is None:
        yolo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'YOLO')
        default_img = "urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg"
        args.image_path = os.path.join(yolo_dir, 'train', 'images', default_img)
    
    # Ensure the image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the image
    image = Image.open(args.image_path)
    image_np = np.array(image)
    
    print(f"Processing image: {os.path.basename(args.image_path)}")
    
    # Detect tree trunks
    trunk_detections = detect_tree_trunks(args.image_path)
    print(f"Detected {len(trunk_detections)} tree trunks")
    
    # Create a simulated ground mask (just for demo purposes)
    # In a real scenario, this would come from the segmentation model
    h, w = image_np.shape[:2]
    bottom_third = int(h * 2/3)
    simulated_ground_mask = np.zeros((h, w), dtype=np.uint8)
    simulated_ground_mask[bottom_third:, :] = 1
    
    # Enhance the ground mask with trunk detections
    enhanced_mask = enhance_ground_mask_with_trunk_detections(
        simulated_ground_mask, trunk_detections, expansion_factor=0.3)
    
    # Create a visualization
    print("Creating visualization grid")
    vis_grid = create_visualization_grid(image_np, trunk_detections, 
                                        simulated_ground_mask, enhanced_mask)
    
    # Save the visualization
    output_grid_path = os.path.join(args.output_dir, 'trunk_detection_demo.jpg')
    cv2.imwrite(output_grid_path, cv2.cvtColor(vis_grid, cv2.COLOR_RGB2BGR))
    print(f"Visualization saved to {output_grid_path}")
    
    # Also save individual component images
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Save image with bounding boxes
    output_boxes_path = os.path.join(args.output_dir, f"{base_name}_boxes.jpg")
    visualize_bboxes(args.image_path, trunk_detections, output_boxes_path)
    print(f"Image with bounding boxes saved to {output_boxes_path}")
    
    # Save ground mask
    output_mask_path = os.path.join(args.output_dir, f"{base_name}_ground_mask.png")
    cv2.imwrite(output_mask_path, simulated_ground_mask * 255)
    
    # Save enhanced mask
    output_enhanced_path = os.path.join(args.output_dir, f"{base_name}_enhanced_mask.png")
    cv2.imwrite(output_enhanced_path, enhanced_mask * 255)
    
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
