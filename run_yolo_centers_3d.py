#!/usr/bin/env python3
"""
Example YOLO Centers to 3D Visualization

This script runs the YOLO centers to 3D visualization with example data.
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run YOLO centers to 3D visualization with example data")
    
    parser.add_argument("--output", type=str, default="yolo_centers_3d.html",
                        help="Path to save the visualization HTML")
    
    return parser.parse_args()

def find_example_data():
    """Find example data files"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Default paths
    image_path = os.path.join(project_root, "dataset", "input_images", 
                             "urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg")
    label_path = os.path.join(project_root, "YOLO", "train", "labels",
                             "urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.txt")
    depth_path = os.path.join(project_root, "outputs", 
                             "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab", "depth_map.png")
    ply_path = os.path.join(project_root, "outputs", 
                           "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab", "point_cloud.ply")
    
    # Check if files exist
    files_exist = True
    if not os.path.exists(image_path):
        print(f"Warning: Default image not found at {image_path}")
        files_exist = False
    
    if not os.path.exists(label_path):
        print(f"Warning: Default label not found at {label_path}")
        files_exist = False
    
    if not os.path.exists(depth_path):
        print(f"Warning: Default depth map not found at {depth_path}")
        files_exist = False
    
    if not os.path.exists(ply_path):
        print(f"Warning: Default point cloud not found at {ply_path}")
        files_exist = False
    
    if not files_exist:
        # Try to find alternative files
        print("Searching for alternative example files...")
        
        # Try to find an image
        if not os.path.exists(image_path):
            for root, dirs, files in os.walk(os.path.join(project_root, "dataset")):
                for file in files:
                    if file.endswith((".jpg", ".jpeg", ".png")):
                        image_path = os.path.join(root, file)
                        print(f"Found alternative image: {image_path}")
                        break
                if os.path.exists(image_path):
                    break
        
        # Try to find a YOLO label
        if not os.path.exists(label_path):
            for root, dirs, files in os.walk(os.path.join(project_root, "YOLO")):
                for file in files:
                    if file.endswith(".txt") and "classes" not in file:
                        label_path = os.path.join(root, file)
                        print(f"Found alternative label: {label_path}")
                        break
                if os.path.exists(label_path):
                    break
        
        # Try to find a depth map
        if not os.path.exists(depth_path):
            for root, dirs, files in os.walk(os.path.join(project_root, "outputs")):
                for file in files:
                    if "depth" in file.lower() and file.endswith(".png"):
                        depth_path = os.path.join(root, file)
                        print(f"Found alternative depth map: {depth_path}")
                        break
                if os.path.exists(depth_path):
                    break
        
        # Try to find a point cloud
        if not os.path.exists(ply_path):
            for root, dirs, files in os.walk(os.path.join(project_root, "outputs")):
                for file in files:
                    if file.endswith(".ply"):
                        ply_path = os.path.join(root, file)
                        print(f"Found alternative point cloud: {ply_path}")
                        break
                if os.path.exists(ply_path):
                    break
    
    # Check if we found all required files
    if not os.path.exists(image_path) or not os.path.exists(label_path) or \
       not os.path.exists(depth_path) or not os.path.exists(ply_path):
        print("Error: Could not find all required example files.")
        return None, None, None, None
    
    return image_path, label_path, depth_path, ply_path

def main():
    """Main function"""
    args = parse_args()
    
    # Find example data
    print("Finding example data files...")
    image_path, label_path, depth_path, ply_path = find_example_data()
    
    if not image_path:
        print("Error: Could not find example data. Exiting.")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_centers_to_3d.py"),
        f"--image={image_path}",
        f"--label={label_path}",
        f"--depth={depth_path}",
        f"--ply={ply_path}",
        f"--output={args.output}"
    ]
    
    # Run visualization
    print("Running visualization...")
    subprocess.run(cmd)
    
    print(f"Visualization saved to: {args.output}")

if __name__ == "__main__":
    main()
