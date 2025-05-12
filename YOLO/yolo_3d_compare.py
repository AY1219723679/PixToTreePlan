#!/usr/bin/env python3
"""
YOLO Boxes to 3D Points Comparison Demo

This script demonstrates how to:
1. Load YOLO bounding box labels
2. Convert them to pixel coordinates
3. Extract center points from these regions
4. Convert to 3D coordinates using two different depth maps
5. Visualize the results in a side-by-side comparison

Usage:
    python yolo_3d_compare.py --image <path_to_image> --label <path_to_label> 
                             --depth <path_to_depth_map> --depth_masked <path_to_masked_depth_map>
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Import functions from the visualization module
from side_by_side_viz import visualize_point_clouds_side_by_side

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Try to import the coordinate utilities with multiple fallbacks
try:
    # First attempt - direct import
    from main.img_to_pointcloud.coord_utils import pixel_coords_to_3d, load_depth_map
    print("Successfully imported from main.img_to_pointcloud")
except ImportError:
    try:
        # Second attempt - via core symlink
        from core.main.img_to_pointcloud.coord_utils import pixel_coords_to_3d, load_depth_map
        print("Successfully imported from core.main.img_to_pointcloud")
    except ImportError:
        # This is a severe error - we can't continue without these imports
        print("ERROR: Cannot import coordinate utilities after multiple attempts.")
        print("Please run the VS Code task 'Setup Symbolic Link' or manually run:")
        print("New-Item -ItemType Junction -Path \"core/main\" -Target \"main\" -Force")
        sys.exit(1)

# Import YOLO utilities (should be local to the script)
try:
    from yolo_utils import load_yolo_bboxes
except ImportError:
    print("ERROR: Cannot import YOLO utilities.")
    print("Make sure you're running the script from the YOLO directory.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare 3D points from different depth maps using YOLO boxes")
    
    parser.add_argument(
        "--image", 
        type=str, 
        default=None,
        help="Path to the image file"
    )
    
    parser.add_argument(
        "--label", 
        type=str, 
        default=None,
        help="Path to the YOLO label file (.txt)"
    )
    
    parser.add_argument(
        "--depth", 
        type=str, 
        default=None,
        help="Path to the original depth map file"
    )
    
    parser.add_argument(
        "--depth_masked", 
        type=str, 
        default=None,
        help="Path to the masked depth map file (ground-only)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="yolo_3d_output",
        help="Directory to save the output files"
    )
    
    parser.add_argument(
        "--z_scale", 
        type=float, 
        default=0.5,
        help="Scale factor for depth values"
    )
    
    parser.add_argument(
        "--auto_ground", 
        action="store_true",
        help="Automatically generate ground-only depth map if not provided"
    )
    
    parser.add_argument(
        "--skip_validation", 
        action="store_true",
        help="Skip validation of depth maps against image dimensions"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force continue even if validation fails (no prompts)"
    )
    
    return parser.parse_args()


def sample_points_from_boxes(bboxes):
    """
    Get center points from YOLO bounding boxes.
    
    Args:
        bboxes (list): List of bounding box dictionaries from load_yolo_bboxes
        
    Returns:
        np.ndarray: Array of shape (N, 2) containing center points of each box
    """
    center_points = []
    
    for bbox in bboxes:
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']
        
        # Compute the geometric center of the bounding box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        # Add the center point
        center_points.append([x_center, y_center])
    
    # Convert to numpy array
    if center_points:
        return np.array(center_points)
    else:
        return np.array([])


def visualize_3d_popup(coords_3d, title="3D Point Cloud Visualization"):
    """
    Create a popup window for interactive 3D visualization of point cloud.
    
    Args:
        coords_3d (np.ndarray): 3D coordinates as a numpy array of shape (N, 3)
        title (str): Title for the visualization window
    """
    # Create a separate figure for the popup visualization
    popup_fig = plt.figure(figsize=(10, 8))
    popup_ax = popup_fig.add_subplot(111, projection='3d')
    
    # Color the points by their depth value (Z coordinate)
    norm = plt.Normalize(coords_3d[:, 2].min(), coords_3d[:, 2].max())
    colors = plt.cm.plasma(norm(coords_3d[:, 2]))
    
    # Create the 3D scatter plot
    popup_ax.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        c=colors, 
        s=30,  # Larger point size for better visibility
        alpha=0.8
    )
    
    # Set labels and title
    popup_ax.set_xlabel('X (pixel)')
    popup_ax.set_ylabel('Y (pixel)')
    popup_ax.set_zlabel('Z (depth)')
    popup_ax.set_title(title)
    
    # Add a color bar to show the depth scale
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap='plasma')
    scalar_map.set_array([])
    cbar = popup_fig.colorbar(scalar_map, ax=popup_ax, label='Depth')
    
    # Make the visualization more interactive
    popup_ax.view_init(elev=30, azim=45)  # Set initial viewing angle
    
    # Show the figure in a non-blocking way
    plt.show(block=False)
    
    return popup_fig


def find_matching_output_folder(image_path, outputs_dir):
    """
    Find the output folder that best matches the input image.
    
    Args:
        image_path (str): Path to the input image
        outputs_dir (str): Path to the outputs directory
        
    Returns:
        str: Path to the matching folder, or None if no match found
    """
    if not os.path.exists(outputs_dir):
        print(f"Error: Outputs directory not found at {outputs_dir}")
        return None
    
    img_name = os.path.basename(image_path)
    base_name = os.path.splitext(img_name)[0]
    
    # Parse the image name to extract key components
    name_parts = {}
    
    # Handle RF code format (e.g., urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab)
    if '.rf.' in base_name:
        main_name, rf_code = base_name.split('.rf.')
        name_parts['main'] = main_name
        name_parts['rf_code'] = rf_code
        
    # Handle _jpg_rf_ format (e.g., urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab)
    elif '_jpg_rf_' in base_name:
        main_name, rf_code = base_name.split('_jpg_rf_')
        name_parts['main'] = main_name
        name_parts['rf_code'] = rf_code
    
    # Clean name if needed
    else:
        name_parts['main'] = base_name
        name_parts['rf_code'] = ""
    
    # Extract the image number if available
    # Find the last number in the main name (e.g., extract '33' from 'urban_tree_33')
    parts = name_parts['main'].split('_')
    image_num = None
    for part in reversed(parts):
        if part.isdigit():
            image_num = part
            break
    
    name_parts['image_num'] = image_num
    
    print(f"Looking for output folder for: {img_name}")
    print(f"Main name: {name_parts['main']}")
    print(f"RF code: {name_parts['rf_code']}")
    
    # ------ PRIORITY 1: Find folder with exact RF code match ------
    if name_parts['rf_code']:
        for folder in os.listdir(outputs_dir):
            if name_parts['rf_code'] in folder:
                folder_path = os.path.join(outputs_dir, folder)
                if os.path.isdir(folder_path):
                    print(f"Found exact RF code match: {folder}")
                    return folder_path
    
    # ------ PRIORITY 2: Find folder with exact image number and similar name ------
    if name_parts['image_num']:
        for folder in os.listdir(outputs_dir):
            # Check if both the image number and part of the main name are in the folder name
            if (name_parts['image_num'] in folder and 
                any(part in folder.lower() for part in name_parts['main'].lower().split('_') if len(part) > 3)):
                
                folder_path = os.path.join(outputs_dir, folder)
                if os.path.isdir(folder_path):
                    print(f"Found match with image number and name: {folder}")
                    return folder_path
    
    # ------ PRIORITY 3: Find folder with exact main name match ------
    # This looks for folders that contain the main part of the image name (e.g., 'urban_tree_33')
    for folder in os.listdir(outputs_dir):
        if name_parts['main'].lower() in folder.lower():
            folder_path = os.path.join(outputs_dir, folder)
            if os.path.isdir(folder_path):
                print(f"Found main name match: {folder}")
                return folder_path
    
    # ------ PRIORITY 4: Scoring system for partial matches ------
    best_match = None
    best_score = 0
    
    for folder in os.listdir(outputs_dir):
        score = 0
        folder_lower = folder.lower()
        
        # Check for main name components
        for part in name_parts['main'].lower().split('_'):
            if len(part) > 2 and part in folder_lower:  # Only count meaningful parts
                score += 5
                
        # Bonus for image number match
        if name_parts['image_num'] and name_parts['image_num'] in folder:
            score += 15
            
        # Small bonus for any RF code characters matching
        if name_parts['rf_code'] and name_parts['rf_code'][:4] in folder:
            score += 2
            
        if score > best_score:
            best_score = score
            best_match = os.path.join(outputs_dir, folder)
    
    if best_match:
        print(f"Found best matching folder: {os.path.basename(best_match)}")
        return best_match
    
    return None


def find_default_files(args):
    """
    Find default input files if not provided as arguments.
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        argparse.Namespace: Updated arguments with default files if found
    """
    # Get paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolo_dir = os.path.join(project_root, 'YOLO')
    outputs_dir = os.path.join(project_root, 'outputs')
    
    # Find a default image if not provided
    if args.image is None:
        test_images = [
            os.path.join(yolo_dir, 'train', 'images', 'urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg'),
            os.path.join(project_root, 'dataset', 'input_images', 'urban_tree_33.jpg')
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                args.image = img_path
                print(f"Using default image: {args.image}")
                break
    
    # Find a default label file if not provided but image is available
    if args.label is None and args.image is not None:
        # Try to derive label path from image path
        img_name = os.path.basename(args.image)
        base_name = os.path.splitext(img_name)[0]
        
        # First try exact match
        default_label = os.path.join(yolo_dir, 'train', 'labels', f"{base_name}.txt")
        if os.path.exists(default_label):
            args.label = default_label
            print(f"Using default label: {args.label}")
    
    # Look for corresponding depth map in outputs
    if args.depth is None and args.image is not None:
        # Use the improved matching algorithm to find the best folder
        matching_folder = find_matching_output_folder(args.image, outputs_dir)
        
        if matching_folder:
            print(f"Found matching output folder: {matching_folder}")
            
            # Look for depth map in the matching folder
            depth_path = os.path.join(matching_folder, 'depth_map.png')
            if os.path.exists(depth_path):
                args.depth = depth_path
                print(f"Found matching depth map: {args.depth}")
                
                # Also look for masked depth map in the SAME folder
                potential_masked_names = ['depth_map_masked.png', 'depth_masked.png', 'ground_depth.png', 'masked_depth.png']
                for name in potential_masked_names:
                    masked_path = os.path.join(matching_folder, name)
                    if os.path.exists(masked_path):
                        args.depth_masked = masked_path
                        print(f"Found matching masked depth map: {args.depth_masked}")
                        break
        else:
            print("No matching output folder found for the provided image.")
    
    return args


def validate_depth_map_dimensions(image_path, depth_map_path):
    """
    Validates that the depth map dimensions match the image dimensions.
    
    Args:
        image_path (str): Path to the image file
        depth_map_path (str): Path to the depth map file
        
    Returns:
        bool: True if dimensions match or are proportional, False otherwise
    """
    try:
        # Load image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Load depth map
        depth_map = Image.open(depth_map_path)
        depth_width, depth_height = depth_map.size
        
        # Check if dimensions match exactly
        if img_width == depth_width and img_height == depth_height:
            return True
            
        # Check if dimensions are proportional (within 5% tolerance)
        width_ratio = img_width / depth_width
        height_ratio = img_height / depth_height
        ratio_diff = abs(width_ratio - height_ratio)
        
        if ratio_diff < 0.05:
            print(f"Warning: Image and depth map dimensions don't match exactly, but are proportional.")
            print(f"Image: {img_width}x{img_height}, Depth map: {depth_width}x{depth_height}")
            return True
            
        # Dimensions don't match
        print(f"Warning: Image and depth map dimensions don't match and aren't proportional.")
        print(f"Image: {img_width}x{img_height}, Depth map: {depth_width}x{depth_height}")
        return False
        
    except Exception as e:
        print(f"Error validating dimensions: {e}")
        return False


def create_ground_only_depth(depth_map, output_path=None):
    """
    Create a ground-only depth map by applying simple heuristics.
    This is a fallback when a proper masked depth map isn't available.
    
    Args:
        depth_map (np.ndarray): The original depth map
        output_path (str, optional): Path to save the masked depth map
        
    Returns:
        np.ndarray: Ground-only depth map
    """
    # Create a copy of the depth map
    ground_depth = depth_map.copy()
    
    # Apply a simple heuristic: ground is typically in the lower part of the image
    # and has consistent depth values
    height, width = depth_map.shape
    
    # Focus on the bottom third of the image where ground is likely to be
    bottom_third = depth_map[int(height*2/3):, :]
    bottom_third_valid = bottom_third[bottom_third > 0]  # Only consider non-zero pixels
    
    if len(bottom_third_valid) == 0:
        print("Warning: No valid depth values in the bottom part of the image.")
        return ground_depth  # Return original if we can't determine ground
        
    # Find a threshold to separate ground from non-ground objects
    # Ground is typically at a consistent depth, with objects being closer (smaller depth value)
    threshold = np.percentile(bottom_third_valid, 65)  # 65th percentile often works well
    
    # Create mask: keep pixels that are likely to be ground
    # Ground pixels are those with depth greater than the threshold and not too far
    max_ground_depth = threshold * 1.5  # Don't include very far objects
    ground_mask = (depth_map >= threshold) & (depth_map <= max_ground_depth)
    
    # Apply the mask
    ground_depth[~ground_mask] = 0
    
    # Save the masked depth map if requested
    if output_path:
        Image.fromarray((ground_depth * 255).astype(np.uint8)).save(output_path)
        print(f"Created and saved estimated ground-only depth map to {output_path}")
    
    return ground_depth


def save_point_cloud_to_ply(coords_3d, output_path):
    """
    Save 3D coordinates to a PLY file.
    
    Args:
        coords_3d (np.ndarray): 3D coordinates array of shape (N, 3)
        output_path (str): Path to save the PLY file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the PLY file for writing
    with open(output_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(coords_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # Write vertices
        for i in range(len(coords_3d)):
            x, y, z = coords_3d[i]
            f.write(f"{x} {y} {z}\n")
    
    print(f"Point cloud saved to {output_path}")


def main():
    args = parse_args()
    
    # Find default files if not provided
    args = find_default_files(args)
    
    # Ensure we have the required files
    if args.image is None or not os.path.exists(args.image):
        print("Error: Image file not provided or not found.")
        return
    
    if args.label is None or not os.path.exists(args.label):
        print("Error: Label file not provided or not found.")
        return
    
    if args.depth is None or not os.path.exists(args.depth):
        print("Error: Depth map file not provided or not found.")
        return
    
    # Validate depth map matches image dimensions
    if not args.skip_validation:
        if not validate_depth_map_dimensions(args.image, args.depth):
            print("Error: Depth map dimensions do not match the image dimensions.")
            if not args.force:
                user_input = input("Do you want to continue anyway? (y/n): ")
                if user_input.lower() != 'y':
                    print("Exiting.")
                    return
    
    # Handle masked depth map
    if args.depth_masked is None:
        # Try to find ground mask in the same directory as the depth map
        depth_dir = os.path.dirname(args.depth)
        potential_ground_files = [
            os.path.join(depth_dir, 'ground_mask.png'),
            os.path.join(depth_dir, 'ground_only.png'),
            os.path.join(depth_dir, 'segmentation_ground.png'),
            os.path.join(depth_dir, 'segment_ground.png')
        ]
        
        found_ground_mask = False
        for ground_file in potential_ground_files:
            if os.path.exists(ground_file):
                print(f"Found ground mask: {ground_file}")
                print("Creating ground-only depth map using the ground mask...")
                
                # Load the ground mask and original depth map
                try:
                    ground_mask = load_depth_map(ground_file) > 0  # Convert to binary mask
                    original_depth = load_depth_map(args.depth)
                    
                    # Create masked depth map (zero out non-ground areas)
                    masked_depth = original_depth.copy()
                    masked_depth[~ground_mask] = 0
                    
                    # Save the masked depth map temporarily
                    masked_depth_path = os.path.join(depth_dir, 'depth_map_masked.png')
                    Image.fromarray((masked_depth * 255).astype(np.uint8)).save(masked_depth_path)
                    
                    args.depth_masked = masked_depth_path
                    print(f"Created masked depth map: {args.depth_masked}")
                    found_ground_mask = True
                    break
                except Exception as e:
                    print(f"Failed to create masked depth map: {e}")
        
        # If no ground mask is found, create a ground-only depth map using heuristics
        if not found_ground_mask and args.auto_ground:
            print("No ground mask found. Creating ground-only depth map using heuristics...")
            try:
                original_depth = load_depth_map(args.depth)
                masked_depth_path = os.path.join(depth_dir, 'depth_map_masked.png')
                ground_depth = create_ground_only_depth(original_depth)
                Image.fromarray((ground_depth * 255).astype(np.uint8)).save(masked_depth_path)
                args.depth_masked = masked_depth_path
                print(f"Created estimated ground-only depth map: {args.depth_masked}")
            except Exception as e:
                print(f"Failed to create estimated ground-only depth map: {e}")
    
    # If we have a masked depth map, validate it
    if args.depth_masked and not args.skip_validation:
        if not validate_depth_map_dimensions(args.image, args.depth_masked):
            print("Error: Masked depth map dimensions do not match the image dimensions.")
            if not args.force:
                user_input = input("Do you want to continue anyway? (y/n): ")
                if user_input.lower() != 'y':
                    print("Exiting.")
                    return
      # Load YOLO bounding boxes from the label file
    try:
        print(f"Loading YOLO boxes from: {args.label}")
        bboxes = load_yolo_bboxes(args.label, image_path=args.image)
        print(f"Found {len(bboxes)} bounding boxes.")
    except Exception as e:
        print(f"Error loading YOLO boxes: {e}")
        return
    
    # Sample points from the bounding boxes
    try:
        points = sample_points_from_boxes(bboxes)
        if len(points) == 0:
            print("Warning: No points were extracted from bounding boxes.")
    except Exception as e:
        print(f"Error sampling points: {e}")
        return
      # Load the image for visualization
    try:
        print(f"Loading image from: {args.image}")
        # Use PIL instead of OpenCV for better Unicode path support
        pil_img = Image.open(args.image)
        img = np.array(pil_img)
        # PIL loads in RGB, no need to convert
        print(f"Image loaded successfully: {img.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Load the depth map
    try:
        print(f"Loading depth map from: {args.depth}")
        depth_map = load_depth_map(args.depth)
    except Exception as e:
        print(f"Error loading depth map: {e}")
        return
    
    # Convert 2D points to 3D coordinates using the regular depth map
    try:
        print("Converting points to 3D using regular depth map...")
        coords_3d_full = pixel_coords_to_3d(points, depth_map, z_scale=args.z_scale)
    except Exception as e:
        print(f"Error converting to 3D with regular depth map: {e}")
        return
    
    # If masked depth map is provided, also convert points using it
    coords_3d_masked = None
    if args.depth_masked and os.path.exists(args.depth_masked):
        try:
            print(f"Loading masked depth map from: {args.depth_masked}")
            masked_depth_map = load_depth_map(args.depth_masked)
            
            print("Converting points to 3D using masked depth map...")
            coords_3d_masked = pixel_coords_to_3d(points, masked_depth_map, z_scale=args.z_scale)
        except Exception as e:
            print(f"Error converting to 3D with masked depth map: {e}")
    else:
        print("No masked depth map provided or found. Skipping masked depth conversion.")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the point clouds to PLY files
    if len(coords_3d_full) > 0:
        full_ply_path = os.path.join(args.output_dir, 'full_depth_points.ply')
        try:
            save_point_cloud_to_ply(coords_3d_full, full_ply_path)
            print(f"Saved full depth point cloud to: {full_ply_path}")
        except Exception as e:
            print(f"Error saving full depth point cloud: {e}")
    
    if coords_3d_masked is not None and len(coords_3d_masked) > 0:
        masked_ply_path = os.path.join(args.output_dir, 'ground_depth_points.ply')
        try:
            save_point_cloud_to_ply(coords_3d_masked, masked_ply_path)
            print(f"Saved ground depth point cloud to: {masked_ply_path}")
        except Exception as e:
            print(f"Error saving ground depth point cloud: {e}")
    
    # Visualize the results
    try:
        # Create side-by-side visualization
        print("Creating side-by-side visualization...")
        visualize_point_clouds_side_by_side(
            image=img,
            points=points,
            coords_3d_full=coords_3d_full,
            coords_3d_masked=coords_3d_masked,
            output_path=os.path.join(args.output_dir, 'depth_comparison.png')
        )
        print("Side-by-side visualization complete.")
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    # Open popup windows for 3D visualization
    if len(coords_3d_full) > 0:
        try:
            # Visualize full depth point cloud
            print("Creating 3D visualization of full depth point cloud...")
            visualize_3d_popup(coords_3d_full, title="Full Depth Point Cloud")
        except Exception as e:
            print(f"Error creating full depth 3D visualization: {e}")
    
    if coords_3d_masked is not None and len(coords_3d_masked) > 0:
        try:
            # Visualize masked depth point cloud
            print("Creating 3D visualization of ground depth point cloud...")
            visualize_3d_popup(coords_3d_masked, title="Ground Depth Point Cloud")
        except Exception as e:
            print(f"Error creating ground depth 3D visualization: {e}")
    
    print("\nDone! You can close the visualization windows when finished.")
    plt.show()  # Keep the visualization windows open until manually closed


if __name__ == "__main__":
    main()
