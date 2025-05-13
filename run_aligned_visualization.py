#!/usr/bin/env python3
"""
Run the coordinate-aligned visualization script with example data.
This script specifically addresses coordinate mismatch problems
between point clouds and YOLO center points.
"""

import os
import sys
import subprocess

def main():
    # Set up paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Default test case
    outputs_dir = os.path.join(project_root, "outputs")
    default_folder = os.path.join(outputs_dir, "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab")
    
    # Point cloud path
    ply_path = os.path.join(default_folder, "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: Default PLY file not found at {ply_path}")
        sys.exit(1)
    
    # Get related files
    image_path = os.path.join(default_folder, "original.png")
    depth_path = os.path.join(default_folder, "depth_map.png")
    label_path = os.path.join(project_root, "YOLO", "train", "labels", 
                             "urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.txt")
    
    # Check files
    for path, name in [
        (ply_path, "PLY file"),
        (image_path, "Image file"),
        (depth_path, "Depth map"),
        (label_path, "YOLO label")
    ]:
        if os.path.exists(path):
            print(f"Found {name}: {path}")
        else:
            print(f"Warning: {name} not found at {path}")
            if name != "PLY file":  # PLY file is required
                # Try to find an alternative
                if name == "Image file":
                    alt_path = os.path.join(default_folder, "image.png")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        print(f"Using alternative image file: {alt_path}")
                elif name == "Depth map":
                    for alt_name in ["depth_masked.png", "depth.png"]:
                        alt_path = os.path.join(default_folder, alt_name)
                        if os.path.exists(alt_path):
                            depth_path = alt_path
                            print(f"Using alternative depth map: {alt_path}")
                            break
    
    # Output path
    output_path = os.path.join(project_root, "aligned_3d_view.html")
    
    # Build command with large point and center sizes for better visibility
    cmd = [
        sys.executable,
        os.path.join(project_root, "coordinate_aligned_visualization.py"),
        f"--ply={ply_path}",
        f"--image={image_path}",
        f"--depth={depth_path}",
        f"--label={label_path}",
        f"--output={output_path}",
        "--point_size=8.0",      # Large ground point cloud points
        "--center_size=20.0"     # Very large center points
    ]
    
    # Run visualization
    print("\nRunning coordinate-aligned visualization...")
    print("=" * 50)
    subprocess.run(cmd)
    print("=" * 50)
    
    print(f"\nCoordinate-aligned visualization saved to: {output_path}")
    print("\nThis visualization uses a new algorithm that properly aligns the")
    print("point cloud and YOLO detection centers in the same coordinate space.")
    print("\nThe visualization should now show:")
    print("1. Ground point cloud with properly sized points")
    print("2. YOLO detection centers properly placed in 3D space")
    print("3. Vertical guide lines connecting centers to the ground")
    
if __name__ == "__main__":
    main()
