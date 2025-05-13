#!/usr/bin/env python3
"""
Center Point Utilities for 3D Visualization

This module provides utilities for working with center points derived from YOLO bounding boxes
and projecting them into 3D space using depth maps.
"""

import os
import sys
import numpy as np
from PIL import Image

def load_yolo_labels(label_path, image_path=None):
    """
    Load YOLO bounding boxes from a label file.
    
    Args:
        label_path (str): Path to the YOLO label file
        image_path (str, optional): Path to the image file for reference dimensions
        
    Returns:
        list: List of dictionaries with bounding box coordinates
    """
    boxes = []
    
    # If image path is provided, get the dimensions
    img_width, img_height = None, None
    if image_path and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                print(f"Loaded image dimensions: {img_width}x{img_height}")
        except Exception as e:
            print(f"Error loading image dimensions: {e}")
    
    # Read YOLO format labels
    try:
        with open(label_path, "r") as file:
            lines = file.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # class x_center y_center width height
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # YOLO format has normalized coordinates (0-1)
                    # Convert to absolute pixel coordinates if image dimensions are available
                    if img_width and img_height:
                        x1 = int((x_center - width / 2) * img_width)
                        y1 = int((y_center - height / 2) * img_height)
                        x2 = int((x_center + width / 2) * img_width)
                        y2 = int((y_center + height / 2) * img_height)
                    else:
                        # Keep normalized if no image dimensions
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                    
                    boxes.append({
                        "class": class_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1
                    })
        
        print(f"Loaded {len(boxes)} bounding boxes from {label_path}")
        
    except Exception as e:
        print(f"Error loading YOLO labels: {e}")
    
    return boxes


def extract_center_points(boxes):
    """
    Extract 2D center points from bounding boxes.
    
    Args:
        boxes (list): List of bounding box dictionaries
        
    Returns:
        np.ndarray: 2D center point coordinates of shape (N, 2)
    """
    center_points = []
    
    for box in boxes:
        # Calculate center of bounding box
        x_center = (box["x1"] + box["x2"]) / 2
        y_center = (box["y1"] + box["y2"]) / 2
        center_points.append([x_center, y_center])
    
    if center_points:
        return np.array(center_points)
    else:
        return np.array([])


def load_depth_map(path):
    """
    Load a depth map from file.
    
    Args:
        path (str): Path to the depth map file
        
    Returns:
        np.ndarray: Depth map as a numpy array
    """
    try:
        img = Image.open(path)
        depth = np.array(img)
        
        # Normalize if not already in 0-1 range
        if depth.max() > 1.0:
            depth = depth / 255.0
            
        print(f"Loaded depth map with shape: {depth.shape}")
        return depth
    except Exception as e:
        print(f"Error loading depth map: {e}")
        return np.zeros((100, 100))  # Return empty depth map


def project_to_3d(center_points, depth_map, z_scale=0.5):
    """
    Project 2D center points to 3D using a depth map.
    
    Args:
        center_points (np.ndarray): 2D center points of shape (N, 2)
        depth_map (np.ndarray): Depth map as a 2D numpy array
        z_scale (float): Scale factor for depth values
        
    Returns:
        np.ndarray: 3D coordinates of shape (N, 3)
    """
    if len(center_points) == 0:
        return np.array([])
    
    # Check depth map shape and handle RGB depth maps
    if len(depth_map.shape) > 2:
        # Convert to grayscale if it's RGB
        depth_map = depth_map.mean(axis=2)
        
    height, width = depth_map.shape
    points_3d = []
    
    for point in center_points:
        x, y = point
        
        # Ensure coordinates are within bounds
        x_int = int(np.clip(x, 0, width - 1))
        y_int = int(np.clip(y, 0, height - 1))
        
        # Get depth value
        depth = depth_map[y_int, x_int]
        
        # Only include points with valid depth
        if depth > 0:
            # Apply scaling to depth
            z = depth * z_scale
            # Store 3D point (X, Y, Z)
            points_3d.append([x, y, z])
    
    if points_3d:
        return np.array(points_3d)
    else:
        return np.array([])


def load_and_project(label_path, image_path, depth_map_path, z_scale=0.5):
    """
    Complete pipeline to load YOLO boxes and project centers to 3D.
    
    Args:
        label_path (str): Path to the YOLO label file
        image_path (str): Path to the original image
        depth_map_path (str): Path to the depth map
        z_scale (float): Scale factor for depth values
        
    Returns:
        np.ndarray: 3D center points coordinates
    """
    print(f"Loading YOLO labels from: {label_path}")
    boxes = load_yolo_labels(label_path, image_path)
    
    print("Extracting center points...")
    center_points_2d = extract_center_points(boxes)
    
    print(f"Loading depth map from: {depth_map_path}")
    depth_map = load_depth_map(depth_map_path)
    
    print("Projecting center points to 3D...")
    center_points_3d = project_to_3d(center_points_2d, depth_map, z_scale)
    
    return center_points_3d
