#!/usr/bin/env python3
"""
YOLO Boxes to 3D Points Demo

This script demonstrates how to:
1. Load YOLO bounding box labels
2. Convert them to pixel coordinates
3. Sample points from these regions
4. Convert to 3D coordinates using a depth map
5. Visualize the results

Usage:
    python yolo_to_3d_demo.py --image <path_to_image> --label <path_to_label> --depth <path_to_depth_map>
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Use the import helper to set up the correct paths
try:
    from import_helpers import setup_imports
    if not setup_imports():
        print("Failed to set up imports correctly. Creating symbolic link manually...")
        # Get the project root and create the symbolic link
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_path = os.path.join(project_root, 'main')
        core_dir = os.path.join(project_root, 'core')
        core_main_path = os.path.join(core_dir, 'main')
        
        # Make sure the core directory exists
        if not os.path.exists(core_dir):
            os.makedirs(core_dir, exist_ok=True)
            
        # Create the symbolic link
        if os.name == 'nt':  # Windows
            os.system(f'powershell -Command "New-Item -ItemType Junction -Path \\"{core_main_path}\\" -Target \\"{main_path}\\" -Force"')
        else:
            os.symlink(main_path, core_main_path, target_is_directory=True)
except ImportError:
    # If import_helpers.py doesn't exist, set up the path manually
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    # Check if we need to create the symbolic link
    main_path = os.path.join(project_root, 'main')
    core_main_path = os.path.join(project_root, 'core', 'main')
    
    if not os.path.exists(core_main_path) and os.path.exists(main_path):
        print("Creating symbolic link for core/main...")
        core_dir = os.path.join(project_root, 'core')
        if not os.path.exists(core_dir):
            os.makedirs(core_dir, exist_ok=True)
            
        if os.name == 'nt':  # Windows
            os.system(f'powershell -Command "New-Item -ItemType Junction -Path \\"{core_main_path}\\" -Target \\"{main_path}\\" -Force"')
        else:
            os.symlink(main_path, core_main_path, target_is_directory=True)

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
    parser = argparse.ArgumentParser(description="Convert YOLO boxes to 3D points using depth maps")
    
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
        help="Path to the depth map file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="yolo_3d_output",
        help="Directory to save the output files"
    )
    
    parser.add_argument(
        "--sample_points", 
        type=int, 
        default=100,
        help="Number of points to sample per bounding box"
    )
    
    parser.add_argument(
        "--z_scale", 
        type=float, 
        default=0.5,
        help="Scale factor for depth values"
    )
    
    return parser.parse_args()


def sample_points_from_boxes(bboxes, num_points_per_box=100):
    """
    Sample points from YOLO bounding boxes.
    
    Args:
        bboxes (list): List of bounding box dictionaries from load_yolo_bboxes
        num_points_per_box (int, optional): Number of points to sample per box. Defaults to 100.
        
    Returns:
        np.ndarray: Array of shape (N, 2) containing sampled points
    """
    all_points = []
    
    for bbox in bboxes:
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']
        
        # Sample points within the bounding box
        x_points = np.random.uniform(x1, x2, num_points_per_box)
        y_points = np.random.uniform(y1, y2, num_points_per_box)
        
        # Combine into (x, y) pairs
        box_points = np.column_stack((x_points, y_points))
        all_points.append(box_points)
    
    # Combine all points into a single array
    if all_points:
        return np.vstack(all_points)
    else:
        return np.array([])


def create_simulated_depth_map(image_path, output_dir):
    """
    Create a simulated depth map from an image when a real one is not available.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save the simulated depth map
        
    Returns:
        str: Path to the created depth map or None on failure
    """
    try:
        try:
            # First try with PIL (better Unicode path support)
            from PIL import Image, ImageFilter
            import numpy as np
            
            # Open the image with PIL
            with Image.open(image_path) as img_pil:
                img_gray = img_pil.convert('L')
                
                # Convert to numpy array for processing
                img = np.array(img_gray)
                
                # Apply Gaussian blur for smoothing (using PIL)
                img_blur = img_gray.filter(ImageFilter.GaussianBlur(radius=7))
                sim_depth = np.array(img_blur)
        except Exception as pil_error:
            print(f"PIL processing failed: {pil_error}, falling back to OpenCV")
            # Fall back to OpenCV if PIL fails
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Could not read the image with OpenCV")
                
            # Add some Gaussian blur to smooth it
            sim_depth = cv2.GaussianBlur(img, (15, 15), 0)
        
        # Apply a simple gradient overlay (make bottom parts closer)
        h, w = sim_depth.shape
        gradient = np.zeros_like(sim_depth, dtype=np.float32)
        for y in range(h):
            gradient[y, :] = y / h  # Simple top-to-bottom gradient
        
        # Combine with the image (weighted average)
        sim_depth = cv2.addWeighted(sim_depth, 0.7, (gradient * 255).astype(np.uint8), 0.3, 0)
        
        # Save to a file in the output directory
        os.makedirs(output_dir, exist_ok=True)
        sim_depth_path = os.path.join(output_dir, "simulated_depth.png")
        
        # Try saving with PIL first for better Unicode support
        try:
            Image.fromarray(sim_depth).save(sim_depth_path)
        except Exception:
            # Fall back to OpenCV
            cv2.imwrite(sim_depth_path, sim_depth)
        
        # Verify the file was actually created
        if os.path.exists(sim_depth_path):
            return sim_depth_path
        else:
            raise FileNotFoundError(f"Failed to save simulated depth map to {sim_depth_path}")
            
    except Exception as e:
        print(f"Failed to create simulated depth map: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Find a default image if not provided
    if args.image is None:
        # First try the YOLO train images directory
        default_img = os.path.join(yolo_dir, 'train', 'images', 'urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg')
        
        # If not found, try to find any image in the YOLO train directory
        if not os.path.exists(default_img):
            images_dir = os.path.join(yolo_dir, 'train', 'images')
            if os.path.exists(images_dir):
                for img_file in os.listdir(images_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        default_img = os.path.join(images_dir, img_file)
                        break
        
        # If still not found, try the dataset directory
        if not os.path.exists(default_img):
            dataset_dir = os.path.join(project_root, 'dataset', 'input_images')
            if os.path.exists(dataset_dir):
                for img_file in os.listdir(dataset_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        default_img = os.path.join(dataset_dir, img_file)
                        break
        
        if os.path.exists(default_img):
            args.image = default_img
            print(f"Using default image: {args.image}")
    
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
        else:
            # If no exact match, just try to find a matching label file
            labels_dir = os.path.join(yolo_dir, 'train', 'labels')
            if os.path.exists(labels_dir):
                for label_file in os.listdir(labels_dir):
                    if label_file.lower().endswith('.txt'):
                        args.label = os.path.join(labels_dir, label_file)
                        print(f"Using label file: {label_file}")
                        break
      # Find a default depth map if not provided but image is available
    if args.depth is None and args.image is not None:
        # Try to find a corresponding depth map in outputs
        img_name = os.path.basename(args.image)
        base_name = os.path.splitext(img_name)[0]
        outputs_dir = os.path.join(project_root, 'outputs')
        depth_map_found = False
        
        # Only try to search if the outputs directory exists
        if os.path.exists(outputs_dir):
            try:
                # Check for exact directory match
                img_output_dir = os.path.join(outputs_dir, base_name)
                if os.path.exists(img_output_dir):
                    default_depth = os.path.join(img_output_dir, 'depth_map.png')
                    if os.path.exists(default_depth):
                        args.depth = default_depth
                        print(f"Using default depth map: {args.depth}")
                        depth_map_found = True
                
                # If still no depth map, look for any depth map
                if not depth_map_found:
                    # Try to find any output directory with a depth map
                    for output_dir in os.listdir(outputs_dir):
                        try:
                            output_path = os.path.join(outputs_dir, output_dir)
                            if os.path.isdir(output_path):
                                depth_path = os.path.join(output_path, 'depth_map.png')
                                if os.path.exists(depth_path):
                                    # Quick test to make sure we can open this file
                                    try:
                                        with open(depth_path, 'rb') as test_file:
                                            # Just testing if we can open it
                                            pass
                                        args.depth = depth_path
                                        print(f"Using depth map: {args.depth}")
                                        depth_map_found = True
                                        break
                                    except:
                                        print(f"Found depth map at {depth_path} but couldn't open it, skipping")
                        except Exception as e:
                            # Continue if a specific directory causes an error
                            print(f"Error checking output directory: {e}")
                            continue
            except Exception as e:
                print(f"Error searching for depth maps: {e}")
    
    return args


def main():
    args = parse_args()
    
    # Find default files if not provided
    args = find_default_files(args)
    
    # Ensure we have an image file
    if args.image is None or not os.path.exists(args.image):
        print("Error: Image file not provided or not found.")
        return
    
    # Ensure we have a label file
    if args.label is None or not os.path.exists(args.label):
        print("Error: Label file not provided or not found.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
      # Handle missing depth map or if path contains Unicode characters
    create_simulated = False
    
    if args.depth is None:
        create_simulated = True
    elif not os.path.exists(args.depth):
        print(f"Warning: Depth map not found at: {args.depth}")
        create_simulated = True
    else:
        try:
            # Test if we can actually open the file (to catch encoding issues)
            with open(args.depth, 'rb') as f:
                # Just check if we can open it
                pass
        except Exception as e:
            print(f"Warning: Cannot open depth map due to encoding issue: {e}")
            create_simulated = True
    
    if create_simulated:
        print("Creating a simple simulated depth map from the input image.")
        # Create a simulated depth map from the image
        sim_depth_path = create_simulated_depth_map(args.image, args.output_dir)
        if sim_depth_path and os.path.exists(sim_depth_path):
            args.depth = sim_depth_path
            print(f"Created simulated depth map: {args.depth}")
        else:
            print("Error: No depth map available. Exiting.")
            return
    
    # Print information about the files being used
    print(f"Processing image: {os.path.basename(args.image)}")
    print(f"Using label file: {os.path.basename(args.label)}")
    print(f"Using depth map: {os.path.basename(args.depth)}")
    
    # Load YOLO bounding boxes
    bboxes = load_yolo_bboxes(args.label, args.image)
    print(f"Loaded {len(bboxes)} bounding boxes")
    
    # Load the depth map
    depth_map = load_depth_map(args.depth)
    print(f"Loaded depth map of shape {depth_map.shape}")
    
    # Sample points from bounding boxes
    sampled_points = sample_points_from_boxes(bboxes, args.sample_points)
    print(f"Sampled {len(sampled_points)} points from bounding boxes")
    
    if len(sampled_points) == 0:
        print("No valid points to process. Exiting.")
        return
    
    # Convert to 3D coordinates
    coords_3d = pixel_coords_to_3d(sampled_points, depth_map, z_scale=args.z_scale)
    print(f"Converted points to 3D coordinates")
    
    # Get normalized 3D coordinates for visualization
    coords_3d_norm = pixel_coords_to_3d(sampled_points, depth_map, z_scale=args.z_scale, normalize=True)
      # Load and prepare the image for visualization using PIL (better Unicode support)
    try:
        from PIL import Image
        import numpy as np
        
        # Load with PIL and convert to numpy array in RGB format
        img = np.array(Image.open(args.image).convert('RGB'))
    except Exception as e:
        print(f"Failed to load image with PIL: {e}, falling back to OpenCV")
        # Fall back to OpenCV
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
    
    ax1.scatter(sampled_points[:, 0], sampled_points[:, 1], color='yellow', s=1, alpha=0.5)
    ax1.set_title('Image with YOLO Boxes and Sampled Points')
    
    # 2. Depth map with sampled points
    ax2 = fig.add_subplot(132)
    ax2.imshow(depth_map, cmap='plasma')
    ax2.scatter(sampled_points[:, 0], sampled_points[:, 1], color='white', s=1, alpha=0.5)
    ax2.set_title('Depth Map with Sampled Points')
    
    # 3. 3D visualization
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Color the points by their depth value
    colors = plt.cm.plasma(coords_3d_norm[:, 2])
    
    ax3.scatter(
        coords_3d_norm[:, 0], 
        coords_3d_norm[:, 1], 
        coords_3d_norm[:, 2], 
        c=colors, s=5, alpha=0.8
    )
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z (Depth)')
    ax3.set_title('3D Visualization of YOLO Boxes')
    
    # Save the visualization
    output_path = os.path.join(args.output_dir, 'yolo_to_3d_visualization.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Visualization saved to {output_path}")
    
    # Optional: Save the 3D coordinates as a numpy array
    coords_path = os.path.join(args.output_dir, 'yolo_3d_coords.npy')
    np.save(coords_path, coords_3d)
    print(f"3D coordinates saved to {coords_path}")
    
    # Optional: Create a simple PLY file with the 3D points
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_3d_norm)
        
        # Create some default colors based on depth
        colors = np.zeros((len(coords_3d_norm), 3))
        normalized_z = (coords_3d_norm[:, 2] - coords_3d_norm[:, 2].min()) / (coords_3d_norm[:, 2].max() - coords_3d_norm[:, 2].min())
        colors[:, 0] = 1 - normalized_z  # Red channel (higher for smaller depth)
        colors[:, 2] = normalized_z      # Blue channel (higher for greater depth)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        ply_path = os.path.join(args.output_dir, 'yolo_3d_points.ply')
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"3D point cloud saved to {ply_path}")
        
    except ImportError:
        print("Open3D not available. Skipping PLY file creation.")


if __name__ == "__main__":
    main()
