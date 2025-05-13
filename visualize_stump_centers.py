#!/usr/bin/env python3
"""
3D Stump Center Points Visualization

This script visualizes:
1. A ground point cloud from a .ply file10
2. 3D stump center points derived from YOLO bounding boxes

Both sets are displayed in an interactive 3D plot using Matplotlib.
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the project root to the Python path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Try to import the coordinate utilities with fallback
try:
    from main.img_to_pointcloud.coord_utils import pixel_coords_to_3d, load_depth_map
    print("Using coordinate utilities from main module")
except ImportError:
    try:
        from core.main.img_to_pointcloud.coord_utils import pixel_coords_to_3d, load_depth_map
        print("Using coordinate utilities from core.main module")
    except ImportError:
        try:
            # Use our local fallback implementation
            from center_point_utils import project_to_3d as pixel_coords_to_3d
            print("Using local center_point_utils for 3D projection")
            
            # Basic depth map loading function
            def load_depth_map(path):
                """Load depth map from file."""
                import numpy as np
                from PIL import Image
                try:
                    img = Image.open(path)
                    depth = np.array(img) / 255.0  # Normalize to 0-1
                    return depth
                except Exception as e:
                    print(f"Error loading depth map: {e}")
                    return np.zeros((1, 1))  # Return empty depth map
            
            print("Using basic depth map loading function")
        except ImportError:
            print("ERROR: Cannot import coordinate utilities.")
            print("Please run the VS Code task 'Setup Symbolic Link' or manually run:")
            print("New-Item -ItemType Junction -Path \"core/main\" -Target \"main\" -Force")
            sys.exit(1)

# Import YOLO utilities or our local fallback
try:
    from YOLO.yolo_utils import load_yolo_bboxes
    print("Using YOLO utilities from YOLO module")
except ImportError:
    try:
        # Use our local fallback implementation
        from center_point_utils import load_yolo_labels as load_yolo_bboxes
        print("Using local center_point_utils module for YOLO box loading")
    except ImportError:
        print("Warning: YOLO utilities not found. Center point extraction from YOLO boxes will not be available.")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 3D stump center points on ground point cloud")
    
    parser.add_argument(
        "--ply", 
        type=str,
        default="outputs/point_cloud.ply",
        help="Path to the point cloud PLY file"
    )
    
    parser.add_argument(
        "--image", 
        type=str, 
        default=None,
        help="Path to the original image (for reference)"
    )
    
    parser.add_argument(
        "--label", 
        type=str, 
        default=None,
        help="Path to the YOLO label file (.txt) with bounding boxes"
    )
    
    parser.add_argument(
        "--depth", 
        type=str, 
        default=None,
        help="Path to the depth map file (needed to project YOLO centers to 3D)"
    )
    
    parser.add_argument(
        "--z_scale", 
        type=float, 
        default=0.5,
        help="Scale factor for depth values"
    )
    
    parser.add_argument(
        "--title", 
        type=str, 
        default="3D Stump Center Points on Ground Point Cloud",
        help="Title for the plot"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path to save the plot as an image file (optional)"
    )
    
    parser.add_argument(
        "--center_size", 
        type=float, 
        default=50.0,
        help="Size of the center points markers"
    )
    
    parser.add_argument(
        "--point_size", 
        type=float, 
        default=1.0,
        help="Size of the point cloud points markers"
    )
    
    return parser.parse_args()


def load_point_cloud(ply_path):
    """
    Load a point cloud from a PLY file.
    
    Args:
        ply_path (str): Path to the PLY file
        
    Returns:
        tuple: (points, colors) where:
            - points is a numpy array of shape (N, 3)
            - colors is a numpy array of shape (N, 3) with RGB values in range [0, 1]
    """
    print(f"Loading point cloud from: {ply_path}")
    
    try:
        # Load the point cloud using Open3D
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # Extract points and colors
        points = np.asarray(pcd.points)
        
        # Check if colors are available
        if len(pcd.colors) > 0:
            colors = np.asarray(pcd.colors)
        else:
            # If no colors are available, use a default color (gray)
            print("No colors found in PLY file, using default gray.")
            colors = np.ones((len(points), 3)) * 0.5
        
        print(f"Loaded point cloud with {len(points)} points")
        return points, colors
        
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        sys.exit(1)


def extract_center_points_from_yolo(label_path, image_path, depth_map_path, z_scale=0.5):
    """
    Extract 3D center points from YOLO bounding boxes using a depth map.
    
    Args:
        label_path (str): Path to the YOLO label file
        image_path (str): Path to the image file
        depth_map_path (str): Path to the depth map file
        z_scale (float): Scale factor for depth values
        
    Returns:
        np.ndarray: 3D coordinates of shape (N, 3)
    """
    print(f"Extracting center points from YOLO boxes in: {label_path}")
    
    try:
        # Load YOLO bounding boxes
        boxes = load_yolo_bboxes(label_path, image_path=image_path)
        
        # Extract center points from boxes
        center_points = []
        for box in boxes:
            # Calculate the center of the bounding box
            x_center = (box['x1'] + box['x2']) / 2
            y_center = (box['y1'] + box['y2']) / 2
            center_points.append([x_center, y_center])
            
        if len(center_points) == 0:
            print("No bounding boxes found in the label file.")
            return np.array([])
            
        center_points = np.array(center_points)
        print(f"Found {len(center_points)} center points from YOLO boxes")
        
        # Load depth map
        depth_map = load_depth_map(depth_map_path)
        
        # Convert 2D center points to 3D coordinates using depth map
        centers_3d = pixel_coords_to_3d(center_points, depth_map, z_scale=z_scale)
        
        return centers_3d
        
    except Exception as e:
        print(f"Error extracting center points: {e}")
        return np.array([])


def create_matplotlib_visualization(point_cloud_points, point_cloud_colors, 
                                 center_points=None, title="3D Visualization",
                                 point_size=1, center_size=50):
    """
    Create a matplotlib 3D visualization with point cloud and center points.
    
    Args:
        point_cloud_points (np.ndarray): Point cloud coordinates of shape (N, 3)
        point_cloud_colors (np.ndarray): Point cloud RGB colors of shape (N, 3)
        center_points (np.ndarray, optional): Center point coordinates of shape (M, 3)
        title (str): Title for the plot
        point_size (float): Size of point cloud points
        center_size (float): Size of center points
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ensure colors are in proper format for matplotlib (0-1 range)
    if np.max(point_cloud_colors) > 1.0:
        point_cloud_colors = point_cloud_colors / 255.0
    
    # Subsample the point cloud if it's very large (for better performance)
    max_points = 10000  # Maximum number of points to display
    if len(point_cloud_points) > max_points:
        print(f"Point cloud has {len(point_cloud_points)} points. Subsampling to {max_points} points for visualization...")
        indices = np.random.choice(len(point_cloud_points), max_points, replace=False)
        point_cloud_points = point_cloud_points[indices]
        point_cloud_colors = point_cloud_colors[indices]
    
    # Plot the ground point cloud
    ax.scatter(
        point_cloud_points[:, 0],
        point_cloud_points[:, 1],
        point_cloud_points[:, 2],
        s=point_size,
        c=point_cloud_colors,
        marker='.',
        label='Ground Point Cloud'
    )
    
    # Add center points if available
    if center_points is not None and len(center_points) > 0:
        ax.scatter(
            center_points[:, 0],
            center_points[:, 1],
            center_points[:, 2],
            s=center_size,
            c='red',
            marker='o',
            label='Stump Centers',
            edgecolor='black'
        )
        
        # Add vertical lines connecting center points to the ground for better visualization
        if True:  # Set to False if you don't want the connecting lines
            for i, (x, y, z) in enumerate(center_points):
                # Find closest point on the ground (minimum z)
                ground_z = np.min(point_cloud_points[:, 2])
                
                # Draw a vertical line
                ax.plot([x, x], [y, y], [z, ground_z], 
                        color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title(title, fontsize=14)
    
    # Add a legend
    ax.legend()
    
    # Set equal aspect ratio for all axes
    # This is important for proper 3D visualization
    max_range = np.array([
        point_cloud_points[:, 0].max() - point_cloud_points[:, 0].min(),
        point_cloud_points[:, 1].max() - point_cloud_points[:, 1].min(),
        point_cloud_points[:, 2].max() - point_cloud_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (point_cloud_points[:, 0].max() + point_cloud_points[:, 0].min()) * 0.5
    mid_y = (point_cloud_points[:, 1].max() + point_cloud_points[:, 1].min()) * 0.5
    mid_z = (point_cloud_points[:, 2].max() + point_cloud_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set a nice initial viewing angle
    ax.view_init(elev=30, azim=45)
    
    return fig, ax


def find_matching_files(ply_path):
    """
    Find matching image, depth, and label files based on the PLY path.
    
    Args:
        ply_path (str): Path to the PLY file
        
    Returns:
        dict: Dictionary with paths to image, depth, and label files
    """
    result = {
        "image": None,
        "depth": None,
        "label": None
    }
    
    # Get the directory containing the PLY file
    ply_dir = os.path.dirname(ply_path)
    ply_dir_name = os.path.basename(ply_dir)
    
    # Look for image file in the same directory
    potential_images = ["original.png", "image.png", "cutout.png", "original.jpg"]
    for img in potential_images:
        img_path = os.path.join(ply_dir, img)
        if os.path.exists(img_path):
            result["image"] = img_path
            print(f"Found matching image: {img_path}")
            break
    
    # If no image found in the same directory, check the dataset directory
    if result["image"] is None:
        dataset_input_dir = os.path.join(project_root, "dataset", "input_images")
        # Try to extract the base image name from the directory name
        # For directory names like "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab"
        if "_jpg_rf_" in ply_dir_name:
            base_name = ply_dir_name.split("_jpg_rf_")[0]
            potential_images = [f"{base_name}.jpg", f"{base_name}.png"]
            
            for img_name in potential_images:
                img_path = os.path.join(dataset_input_dir, img_name)
                if os.path.exists(img_path):
                    result["image"] = img_path
                    print(f"Found matching image in dataset: {img_path}")
                    break
    
    # Look for depth map file in the same directory
    potential_depths = ["depth_map.png", "depth_masked.png", "ground_depth.png", "depth.png"]
    for depth in potential_depths:
        depth_path = os.path.join(ply_dir, depth)
        if os.path.exists(depth_path):
            result["depth"] = depth_path
            print(f"Found matching depth map: {depth_path}")
            break
    
    # Look for YOLO label file
    if result["image"]:
        # Get image name and try to find matching YOLO label
        img_name = os.path.basename(result["image"])
        base_name = os.path.splitext(img_name)[0]
        
        # Check in YOLO directory
        yolo_dirs = [
            os.path.join(project_root, "YOLO", "train", "labels"),
            os.path.join(project_root, "YOLO", "labels")
        ]
        
        # If the directory name has a format like urban_tree_33_jpg_rf_..., try to extract the base name
        if "_jpg_rf_" in ply_dir_name:
            base_img_name = ply_dir_name.split("_jpg_rf_")[0]
            
            # Try to find YOLO label for the base name
            for yolo_dir in yolo_dirs:
                if os.path.exists(yolo_dir):
                    # Try variations of the name
                    label_variations = [
                        f"{base_img_name}.txt",
                        f"{base_img_name}_jpg.txt",
                        f"{base_img_name}_jpg.rf.{ply_dir_name.split('_jpg_rf_')[1]}.txt",
                        f"{ply_dir_name}.txt"
                    ]
                    
                    for label in label_variations:
                        label_path = os.path.join(yolo_dir, label)
                        if os.path.exists(label_path):
                            result["label"] = label_path
                            print(f"Found matching YOLO label: {label_path}")
                            return result
        
        # Try with the image name directly
        for yolo_dir in yolo_dirs:
            if os.path.exists(yolo_dir):
                # Try variations of the name
                label_variations = [
                    f"{base_name}.txt",
                    f"{base_name}.rf.{ply_dir_name.split('_rf_')[1] if '_rf_' in ply_dir_name else ''}.txt"
                ]
                
                for label in label_variations:
                    label_path = os.path.join(yolo_dir, label)
                    if os.path.exists(label_path):
                        result["label"] = label_path
                        print(f"Found matching YOLO label: {label_path}")
                        return result
    
    # If we still haven't found a label, search through all YOLO label files
    # for a file that contains part of the directory name
    yolo_dir = os.path.join(project_root, "YOLO", "train", "labels")
    if result["label"] is None and os.path.exists(yolo_dir):
        for label_file in os.listdir(yolo_dir):
            # Check if any part of the directory name is in the label file name
            # This is a last-resort fuzzy matching approach
            if label_file.endswith(".txt"):
                for part in ply_dir_name.split("_"):
                    if len(part) > 2 and part in label_file:  # Only consider meaningful parts
                        result["label"] = os.path.join(yolo_dir, label_file)
                        print(f"Found potential YOLO label match: {label_file}")
                        return result
    
    return result


def find_all_ply_files():
    """
    Find all PLY files in the outputs directory and its subdirectories.
    
    Returns:
        list: List of paths to PLY files
    """
    outputs_dir = os.path.join(project_root, "outputs")
    ply_files = []
    
    # Walk through all subdirectories in outputs
    for root, dirs, files in os.walk(outputs_dir):
        for file in files:
            if file.endswith(".ply"):
                ply_files.append(os.path.join(root, file))
    
    return ply_files

def select_ply_file():
    """
    Display a list of available PLY files and let the user select one.
    
    Returns:
        str: Path to the selected PLY file
    """
    ply_files = find_all_ply_files()
    
    if not ply_files:
        print("Error: No PLY files found in the outputs directory or its subdirectories.")
        sys.exit(1)
    
    print("\nAvailable PLY files:")
    for i, ply_file in enumerate(ply_files):
        # Get relative path from the project root for cleaner display
        rel_path = os.path.relpath(ply_file, project_root)
        print(f"[{i}] {rel_path}")
    
    # If there's only one file, use it automatically
    if len(ply_files) == 1:
        selected_idx = 0
        print(f"\nAutomatically selected the only available PLY file: {ply_files[selected_idx]}")
    else:
        # Ask user to select a file
        while True:
            try:
                user_input = input("\nEnter the number of the PLY file to visualize (or press Enter for the first one): ")
                if user_input.strip() == "":
                    selected_idx = 0  # Default to the first file
                    break
                selected_idx = int(user_input)
                if 0 <= selected_idx < len(ply_files):
                    break
                else:
                    print(f"Please enter a number between 0 and {len(ply_files)-1}")
            except ValueError:
                print("Please enter a valid number")
    
    return ply_files[selected_idx]

def main():
    args = parse_args()
    
    # If the default PLY path was not overridden by command line arguments,
    # let the user select a PLY file
    if args.ply == "outputs/point_cloud.ply":
        args.ply = select_ply_file()
    
    # Load point cloud
    if not os.path.exists(args.ply):
        print(f"Error: PLY file not found at {args.ply}")
        sys.exit(1)
    
    points, colors = load_point_cloud(args.ply)
    
    # If image, depth, or label paths are not provided, try to find them automatically
    if args.image is None or args.depth is None or args.label is None:
        matching_files = find_matching_files(args.ply)
        
        if args.image is None:
            args.image = matching_files["image"]
            
        if args.depth is None:
            args.depth = matching_files["depth"]
            
        if args.label is None:
            args.label = matching_files["label"]
    
    # Extract center points if possible
    center_points = None
    if args.label and args.image and args.depth:
        if os.path.exists(args.label) and os.path.exists(args.image) and os.path.exists(args.depth):
            center_points = extract_center_points_from_yolo(
                args.label, args.image, args.depth, args.z_scale
            )
        else:
            print("Warning: One or more files for center point extraction not found.")
    else:
        print("Warning: Image, label, or depth map not provided. Center points will not be displayed.")
    
    # Create visualization
    fig, ax = create_matplotlib_visualization(
        points, colors, center_points, 
        title=args.title,
        point_size=args.point_size,
        center_size=args.center_size
    )
    
    # Save to image if requested
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {args.output}")
    
    # Show the interactive plot
    print("Displaying interactive 3D plot... (close the window to exit)")
    print("You can rotate the view using the mouse.")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
