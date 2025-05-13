#!/usr/bin/env python3
"""
Run the ultimate 3D visualization with all available data sources.

This script automatically finds and uses:
1. Point cloud PLY file
2. YOLO detection labels
3. Original image
4. Depth map
5. Optional class names file

It launches the ultimate visualization with optimal display settings.
"""

import os
import sys
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run the ultimate 3D visualization")
    
    parser.add_argument("--folder", type=str, default=None,
                    help="Specific output folder to use (defaults to urban_tree_33)")
    parser.add_argument("--bright", action="store_true",
                    help="Use bright color mode for better visibility")
    parser.add_argument("--point_size", type=float, default=5.0,
                    help="Size of point cloud points (default: 5.0)")
    parser.add_argument("--center_size", type=float, default=15.0,
                    help="Size of center points (default: 15.0)")
    parser.add_argument("--subsample", type=float, default=1.0,
                    help="Subsample point cloud for performance (0.0-1.0)")
                    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Default folder or user specified
    outputs_dir = os.path.join(project_root, "outputs")
    
    if args.folder:
        default_folder = os.path.join(outputs_dir, args.folder)
    else:
        # Try to find the urban_tree_33 example
        default_folder = None
        for folder in os.listdir(outputs_dir):
            if folder.startswith("urban_tree_33"):
                default_folder = os.path.join(outputs_dir, folder)
                break
        
        # If not found, just use the first folder
        if not default_folder:
            try:
                folders = [f for f in os.listdir(outputs_dir) 
                          if os.path.isdir(os.path.join(outputs_dir, f))]
                if folders:
                    default_folder = os.path.join(outputs_dir, folders[0])
            except Exception:
                pass
                
        # If still not found, use hardcoded path
        if not default_folder:
            default_folder = os.path.join(outputs_dir, "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab")
    
    # Point cloud path
    ply_path = os.path.join(default_folder, "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found at {ply_path}")
        sys.exit(1)
    
    # Get related files
    image_path = None
    for img_name in ["original.png", "image.png", "source.png"]:
        test_path = os.path.join(default_folder, img_name)
        if os.path.exists(test_path):
            image_path = test_path
            break
    
    depth_path = None
    for depth_name in ["depth_map.png", "depth_masked.png", "depth.png"]:
        test_path = os.path.join(default_folder, depth_name)
        if os.path.exists(test_path):
            depth_path = test_path
            break
    
    # Extract base filename for YOLO label matching
    base_folder_name = os.path.basename(default_folder)
    label_paths = []
    
    # Try different label locations
    yolo_label_dirs = [
        os.path.join(project_root, "YOLO", "train", "labels"),
        os.path.join(project_root, "YOLO", "labels"),
        os.path.join(project_root, "labels")
    ]
    
    for label_dir in yolo_label_dirs:
        if os.path.exists(label_dir):
            # Try exact match first
            exact_label = os.path.join(label_dir, f"{base_folder_name}.txt")
            if os.path.exists(exact_label):
                label_paths.append(exact_label)
            
            # Try partial matches
            parts = base_folder_name.split("_")
            if len(parts) >= 2:
                base_name = "_".join(parts[:2])  # First two parts
                for label_file in os.listdir(label_dir):
                    if label_file.startswith(base_name) and label_file.endswith(".txt"):
                        label_path = os.path.join(label_dir, label_file)
                        if label_path not in label_paths:
                            label_paths.append(label_path)
    
    label_path = label_paths[0] if label_paths else None
    
    # Look for class names file
    class_names_path = None
    for names_file in ["classes.txt", "class_names.txt", "yolo_classes.txt"]:
        for path in [project_root, os.path.join(project_root, "YOLO")]:
            test_path = os.path.join(path, names_file)
            if os.path.exists(test_path):
                class_names_path = test_path
                break
        if class_names_path:
            break
    
    # Output path
    output_path = os.path.join(project_root, "ultimate_3d_visualization.html")
    
    # Report what we found
    print("\nFound the following resources for visualization:")
    print("=" * 70)
    print(f"Point cloud: {ply_path}")
    if image_path:
        print(f"Image: {image_path}")
    else:
        print("Image: Not found")
    
    if depth_path:
        print(f"Depth map: {depth_path}")
    else:
        print("Depth map: Not found")
    
    if label_path:
        print(f"YOLO label: {label_path}")
    else:
        print("YOLO label: Not found")
    
    if class_names_path:
        print(f"Class names: {class_names_path}")
    else:
        print("Class names: Using defaults")
    print("=" * 70)
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join(project_root, "ultimate_3d_visualization.py"),
        f"--ply={ply_path}",
        f"--output={output_path}",
        f"--point_size={args.point_size}",
        f"--center_size={args.center_size}"
    ]
    
    if args.bright:
        cmd.append("--bright_colors")
    
    if args.subsample < 1.0:
        cmd.append(f"--subsample={args.subsample}")
    
    if image_path:
        cmd.append(f"--image={image_path}")
    
    if depth_path:
        cmd.append(f"--depth={depth_path}")
    
    if label_path:
        cmd.append(f"--label={label_path}")
    
    if class_names_path:
        cmd.append(f"--class_names={class_names_path}")
    
    # Run visualization
    print("\nRunning ultimate 3D visualization...")
    print("=" * 70)
    subprocess.run(cmd)
    
    print(f"\nUltimate 3D visualization saved to: {output_path}")
    print("\nThis visualization includes:")
    print("1. Point cloud with original colors")
    print("2. YOLO detection centers properly placed in 3D space")
    print("3. Vertical guide lines connecting centers to the ground")
    print("4. Class labels and coordinate reference axes")

if __name__ == "__main__":
    main()
