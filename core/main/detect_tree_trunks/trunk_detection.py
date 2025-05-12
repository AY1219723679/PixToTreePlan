"""
Tree Trunk Detection Integration Module

This module integrates YOLO-based tree trunk detection with the PixToTreePlan pipeline.
It provides functions to detect tree trunks in images and use the detections to enhance
ground mask extraction and depth estimation.
"""

import os
import numpy as np
from PIL import Image
import sys

# Add the YOLO directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'YOLO'))

# Import YOLO utilities
try:
    from yolo_utils import load_yolo_bboxes
except ImportError:
    print("Warning: yolo_utils module not found. Tree trunk detection will not be available.")


def find_corresponding_label(image_path, labels_dir):
    """
    Find the corresponding YOLO label file for an image.
    
    Args:
        image_path (str): Path to the image file
        labels_dir (str): Directory containing label files
        
    Returns:
        str: Path to the corresponding label file, or None if not found
    """
    # Get the base filename without extension
    base_name = os.path.basename(image_path)
    base_name = os.path.splitext(base_name)[0]
    
    # Look for a matching label file
    label_path = os.path.join(labels_dir, base_name + '.txt')
    
    if os.path.exists(label_path):
        return label_path
    
    return None


def detect_tree_trunks(image_path, yolo_dir=None):
    """
    Detect tree trunks in an image using pre-existing YOLO labels.
    
    Args:
        image_path (str): Path to the image file
        yolo_dir (str, optional): Path to the YOLO directory. If None, will try to use 'YOLO'
            relative to the project root.
            
    Returns:
        list: List of trunk bounding boxes in pixel coordinates, or empty list if no detections
    """
    # Determine YOLO directory
    if yolo_dir is None:
        # Try to locate the YOLO directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        yolo_dir = os.path.join(project_root, 'YOLO')
    
    if not os.path.exists(yolo_dir):
        print(f"Warning: YOLO directory not found: {yolo_dir}")
        return []
    
    # Look for labels directory
    labels_dir = os.path.join(yolo_dir, 'train', 'labels')
    if not os.path.exists(labels_dir):
        print(f"Warning: YOLO labels directory not found: {labels_dir}")
        return []
    
    # Find the corresponding label file
    label_path = find_corresponding_label(image_path, labels_dir)
    if label_path is None:
        print(f"No YOLO label found for image: {os.path.basename(image_path)}")
        return []
    
    # Load the bounding boxes
    try:
        bboxes = load_yolo_bboxes(label_path, image_path)
        return bboxes
    except Exception as e:
        print(f"Error loading YOLO bounding boxes: {e}")
        return []


def enhance_ground_mask_with_trunk_detections(ground_mask, trunk_detections, expansion_factor=0.2):
    """
    Enhance the ground mask by ensuring areas around detected tree trunks are included.
    
    Args:
        ground_mask (numpy.ndarray): Binary ground mask (0/1)
        trunk_detections (list): List of trunk bounding boxes from detect_tree_trunks
        expansion_factor (float): Factor to expand trunk base area
        
    Returns:
        numpy.ndarray: Enhanced ground mask
    """
    if not trunk_detections:
        return ground_mask
    
    # Create a copy of the mask to avoid modifying the original
    enhanced_mask = ground_mask.copy()
    
    h, w = ground_mask.shape[:2]
    
    # For each detected trunk
    for trunk in trunk_detections:
        # Get the bottom part of the trunk (the "base")
        x1 = max(0, int(trunk['x1']))
        x2 = min(w-1, int(trunk['x2']))
        y2 = min(h-1, int(trunk['y2']))  # Bottom of the trunk
        
        # Calculate the height of the trunk
        trunk_height = trunk['height']
        
        # Use only the bottom portion of the trunk
        base_height = int(trunk_height * 0.2)  # Use bottom 20% of the trunk
        y1 = max(0, y2 - base_height)
        
        # Expand the width of the base area
        width = x2 - x1
        expanded_width = width * (1 + expansion_factor)
        x_expansion = int((expanded_width - width) / 2)
        x1_expanded = max(0, x1 - x_expansion)
        x2_expanded = min(w-1, x2 + x_expansion)
        
        # Set this area to be part of the ground mask
        enhanced_mask[y1:y2+1, x1_expanded:x2_expanded+1] = 1
    
    return enhanced_mask


def create_trunk_mask(image_shape, trunk_detections):
    """
    Create a binary mask highlighting tree trunk locations.
    
    Args:
        image_shape (tuple): Shape of the image (height, width)
        trunk_detections (list): List of trunk bounding boxes from detect_tree_trunks
        
    Returns:
        numpy.ndarray: Binary mask with 1 at trunk locations, 0 elsewhere
    """
    h, w = image_shape[:2]
    trunk_mask = np.zeros((h, w), dtype=np.uint8)
    
    for trunk in trunk_detections:
        x1 = max(0, int(trunk['x1']))
        x2 = min(w-1, int(trunk['x2']))
        y1 = max(0, int(trunk['y1']))
        y2 = min(h-1, int(trunk['y2']))
        
        trunk_mask[y1:y2+1, x1:x2+1] = 1
    
    return trunk_mask
