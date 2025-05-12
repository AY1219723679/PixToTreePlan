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
        img_name = os.path.basename(args.image)
        base_name = os.path.splitext(img_name)[0]
        
        # First, check if the outputs directory exists
        if os.path.exists(outputs_dir):
            print(f"Searching for depth maps in: {outputs_dir}")
            
            # Try to find exact matching output directory first
            exact_match_found = False
            for output_folder in os.listdir(outputs_dir):
                # Print all folder names for debugging
                print(f"  Checking output folder: {output_folder}")
                
                # Check for exact match with base name                # Try to match based on filename content
                # Extract the key parts of the image name (remove file extensions and RF codes)
                img_key = base_name.split('_jpg_rf_')[0] if '_jpg_rf_' in base_name else base_name
                output_key = output_folder.split('_jpg_rf_')[0] if '_jpg_rf_' in output_folder else output_folder
                
                # See if we have a match
                if img_key.lower() == output_key.lower() or base_name.lower() in output_folder.lower():
                    print(f"  Found potential matching folder: {output_folder}")
                    output_dir_path = os.path.join(outputs_dir, output_folder)
                    if os.path.isdir(output_dir_path):
                        # Check for depth_map.png
                        depth_path = os.path.join(output_dir_path, 'depth_map.png')
                        if os.path.exists(depth_path):
                            args.depth = depth_path
                            print(f"Found matching depth map: {args.depth}")
                            
                            # Also look for masked depth map in the same directory
                            potential_masked_names = ['depth_map_masked.png', 'depth_masked.png', 'ground_depth.png', 'masked_depth.png']
                            for name in potential_masked_names:
                                masked_path = os.path.join(output_dir_path, name)
                                if os.path.exists(masked_path):
                                    args.depth_masked = masked_path
                                    print(f"Found matching masked depth map: {args.depth_masked}")
                                    break
                            
                            exact_match_found = True
                            break
            
            # If no exact match found, try looking for any depth map
            if not exact_match_found:
                print("No exact matching folder found. Searching all output folders for depth maps...")
                for output_folder in os.listdir(outputs_dir):
                    output_dir_path = os.path.join(outputs_dir, output_folder)
                    if os.path.isdir(output_dir_path):
                        depth_path = os.path.join(output_dir_path, 'depth_map.png')
                        if os.path.exists(depth_path):
                            args.depth = depth_path
                            print(f"Using depth map from non-matching folder: {args.depth}")
                            
                            # Look for masked depth map in the same folder
                            potential_masked_names = ['depth_map_masked.png', 'depth_masked.png', 'ground_depth.png', 'masked_depth.png']
                            for name in potential_masked_names:
                                masked_path = os.path.join(output_dir_path, name)
                                if os.path.exists(masked_path):
                                    args.depth_masked = masked_path
                                    print(f"Using masked depth map: {args.depth_masked}")
                                    break
                            
                            break
        else:
            print(f"Outputs directory not found at: {outputs_dir}")
    
    return args


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
    
    if args.depth_masked is None:
        # Try to find ground mask in the same directory as the depth map
        if args.depth is not None:
            depth_dir = os.path.dirname(args.depth)
            potential_ground_files = [
                os.path.join(depth_dir, 'ground_mask.png'),
                os.path.join(depth_dir, 'ground_only.png'),
                os.path.join(depth_dir, 'segmentation_ground.png'),
                os.path.join(depth_dir, 'segment_ground.png')
            ]
            
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
                        masked_depth_path = os.path.join(os.path.dirname(args.depth), 'depth_map_masked.png')
                        Image.fromarray((masked_depth * 255).astype(np.uint8)).save(masked_depth_path)
                        
                        args.depth_masked = masked_depth_path
                        print(f"Created masked depth map: {args.depth_masked}")
                        break
                    except Exception as e:
                        print(f"Failed to create masked depth map: {e}")
    
    if args.depth_masked is None or not os.path.exists(args.depth_masked):
        print("Warning: Masked depth map not provided or not found. Only the original depth map will be visualized.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print information about the files being used
    print(f"Processing image: {os.path.basename(args.image)}")
    print(f"Using label file: {os.path.basename(args.label)}")
    print(f"Using depth map: {os.path.basename(args.depth)}")
    if args.depth_masked:
        print(f"Using masked depth map: {os.path.basename(args.depth_masked)}")
    
    # Load YOLO bounding boxes
    bboxes = load_yolo_bboxes(args.label, args.image)
    print(f"Loaded {len(bboxes)} bounding boxes")
    
    # Load the depth maps
    depth_map = load_depth_map(args.depth)
    print(f"Loaded depth map of shape {depth_map.shape}")
    
    depth_map_masked = None
    if args.depth_masked:
        depth_map_masked = load_depth_map(args.depth_masked)
        print(f"Loaded masked depth map of shape {depth_map_masked.shape}")
    
    # Get center points from bounding boxes
    center_points = sample_points_from_boxes(bboxes)
    print(f"Extracted {len(center_points)} center points from bounding boxes")
    
    if len(center_points) == 0:
        print("No valid points to process. Exiting.")
        return
    
    # Convert to 3D coordinates with raw pixel coordinates
    coords_3d_raw = pixel_coords_to_3d(center_points, depth_map, z_scale=args.z_scale, normalize=False)
    print(f"Converted points to 3D coordinates (original depth map)")
    
    coords_3d_masked_raw = None
    if depth_map_masked is not None:
        coords_3d_masked_raw = pixel_coords_to_3d(center_points, depth_map_masked, z_scale=args.z_scale, normalize=False)
        print(f"Converted points to 3D coordinates (masked depth map)")
    
    # Load and prepare the image for visualization
    try:
        img = np.array(Image.open(args.image).convert('RGB'))
    except Exception as e:
        print(f"Failed to load image with PIL: {e}, falling back to OpenCV")
        img = cv2.imread(args.image)
        if img is None:
            raise ValueError(f"Could not read image from {args.image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create visualizations
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Original image with bounding boxes
    ax1 = fig.add_subplot(131)
    ax1.imshow(img)
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    
    ax1.scatter(center_points[:, 0], center_points[:, 1], color='yellow', s=5, alpha=0.9)
    ax1.set_title('Image with YOLO Boxes and Center Points')
      # 2. Depth maps comparison
    if depth_map_masked is not None:
        # Find common min/max for depth values for consistent coloring
        # Ignore zeros in the masked depth map when computing min/max
        depth_masked_nonzero = depth_map_masked[depth_map_masked > 0] if np.any(depth_map_masked > 0) else np.array([0])
        
        vmin = min(np.min(depth_map), np.min(depth_masked_nonzero))
        vmax = max(np.max(depth_map), np.max(depth_masked_nonzero))
        
        # Original depth map
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(depth_map, cmap='plasma', vmin=vmin, vmax=vmax)
        ax2.scatter(center_points[:, 0], center_points[:, 1], color='white', s=5, alpha=0.9)
        ax2.set_title('Original Depth Map')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Masked depth map
        ax3 = fig.add_subplot(133)
        # Create a custom colormap that shows zero values as black
        cmap_masked = plt.cm.plasma.copy()
        cmap_masked.set_bad('black')  # For NaN values
        
        # Convert zeros to NaN for visualization
        depth_map_masked_viz = depth_map_masked.copy().astype(float)
        depth_map_masked_viz[depth_map_masked_viz == 0] = np.nan
        
        im3 = ax3.imshow(depth_map_masked_viz, cmap=cmap_masked, vmin=vmin, vmax=vmax)
        ax3.scatter(center_points[:, 0], center_points[:, 1], color='white', s=5, alpha=0.9)
        ax3.set_title('Ground-Only Depth Map')
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    else:
        # Just show the original depth map
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(depth_map, cmap='plasma')
        ax2.scatter(center_points[:, 0], center_points[:, 1], color='white', s=5, alpha=0.9)
        ax2.set_title('Depth Map with Center Points')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Save the visualization of image and depth maps
    depth_viz_path = os.path.join(args.output_dir, 'depth_maps_comparison.png')
    plt.tight_layout()
    plt.savefig(depth_viz_path, dpi=200)
    print(f"Depth map comparison saved to {depth_viz_path}")
    
    # Create side-by-side visualization of 3D point clouds
    if coords_3d_masked_raw is not None:
        # Create and save the side-by-side visualization
        comparison_fig = visualize_point_clouds_side_by_side(
            coords_3d_raw, 
            coords_3d_masked_raw,
            title1="3D Points from Original Depth Map",
            title2="3D Points from Ground-Only Depth Map"
        )
        
        # Save the comparison visualization
        comparison_path = os.path.join(args.output_dir, 'depth_comparison_3d_visualization.png')
        comparison_fig.savefig(comparison_path, dpi=200)
        print(f"3D point cloud comparison saved to {comparison_path}")
    else:
        # Show only the original point cloud
        popup_fig = visualize_3d_popup(coords_3d_raw, title="3D Points from Original Depth Map")
        
        # Save the visualization
        popup_path = os.path.join(args.output_dir, 'original_3d_visualization.png')
        popup_fig.savefig(popup_path, dpi=200)
        print(f"3D visualization saved to {popup_path}")
    
    # Save the 3D coordinates
    np.save(os.path.join(args.output_dir, 'original_3d_coords.npy'), coords_3d_raw)
    if coords_3d_masked_raw is not None:
        np.save(os.path.join(args.output_dir, 'masked_3d_coords.npy'), coords_3d_masked_raw)
    
    # Optional: Create PLY files for the 3D point clouds
    try:
        import open3d as o3d
        
        # Save original depth point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_3d_raw)
        
        # Create some default colors based on depth
        colors = np.zeros((len(coords_3d_raw), 3))
        normalized_z = (coords_3d_raw[:, 2] - coords_3d_raw[:, 2].min()) / (coords_3d_raw[:, 2].max() - coords_3d_raw[:, 2].min())
        colors[:, 0] = 1 - normalized_z  # Red channel (higher for smaller depth)
        colors[:, 2] = normalized_z      # Blue channel (higher for greater depth)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        ply_path = os.path.join(args.output_dir, 'original_depth_points.ply')
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Original depth point cloud saved to {ply_path}")
        
        # Save masked depth point cloud if available
        if coords_3d_masked_raw is not None:
            pcd_masked = o3d.geometry.PointCloud()
            pcd_masked.points = o3d.utility.Vector3dVector(coords_3d_masked_raw)
            
            # Create colors based on depth for masked point cloud
            colors_masked = np.zeros((len(coords_3d_masked_raw), 3))
            normalized_z_masked = (coords_3d_masked_raw[:, 2] - coords_3d_masked_raw[:, 2].min()) / (coords_3d_masked_raw[:, 2].max() - coords_3d_masked_raw[:, 2].min())
            colors_masked[:, 0] = 1 - normalized_z_masked
            colors_masked[:, 2] = normalized_z_masked
            pcd_masked.colors = o3d.utility.Vector3dVector(colors_masked)
            
            ply_path_masked = os.path.join(args.output_dir, 'ground_only_depth_points.ply')
            o3d.io.write_point_cloud(ply_path_masked, pcd_masked)
            print(f"Ground-only depth point cloud saved to {ply_path_masked}")
        
    except ImportError:
        print("Open3D not available. Skipping PLY file creation.")
    
    # Keep the figures open until closed by the user
    plt.show()


if __name__ == "__main__":
    main()
