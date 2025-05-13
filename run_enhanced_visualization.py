#!/usr/bin/env python3
"""
Run the enhanced visualization script with example data
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
    
    # Output path
    output_path = os.path.join(project_root, "enhanced_3d_view.html")
      # Build command
    cmd = [
        sys.executable,
        os.path.join(project_root, "enhanced_visualization.py"),
        f"--ply={ply_path}",
        f"--image={image_path}",
        f"--depth={depth_path}",
        f"--label={label_path}",
        f"--output={output_path}",
        "--point_size=8.0",     # Increased from 3.0 to 8.0 for better visibility
        "--center_size=15.0",   # Slightly increased from 12.0 to 15.0
        "--point_opacity=1.0"   # Full opacity to ensure points are visible
    ]
    
    # Run visualization
    print("\nRunning enhanced visualization...")
    print("=" * 50)
    subprocess.run(cmd)
    print("=" * 50)
    
    print(f"\nEnhanced visualization saved to: {output_path}")
    print("Please check the visualization - both point cloud and object centers should be visible.")
    print("If you still can't see the point cloud, try these troubleshooting steps:")
    print("1. Check that the ply file has valid data (our debug script showed it does)")
    print("2. Try rotating the 3D view - sometimes the default camera angle might hide points")
    print("3. Try different point sizes or opacity settings by modifying the command parameters")

if __name__ == "__main__":
    main()
