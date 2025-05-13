#!/usr/bin/env python3
"""
Run the maximum visibility point cloud visualization
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
    
    # Output path
    output_path = os.path.join(project_root, "max_visibility_pointcloud.html")
    
    # Build command with large point size
    cmd = [
        sys.executable,
        os.path.join(project_root, "maximum_visibility_pointcloud.py"),
        f"--ply={ply_path}",
        f"--output={output_path}",
        "--point_size=15.0",
        "--bright_colors"
    ]
    
    # Run visualization
    print(f"\nRunning maximum visibility point cloud visualization...")
    print(f"=" * 70)
    print(f"Point cloud: {ply_path}")
    print(f"Output file: {output_path}")
    print(f"Point size: 15.0 (extra large)")
    print(f"Using bright colors for maximum visibility")
    print(f"=" * 70)
    
    subprocess.run(cmd)
    
    print(f"\nMaximum visibility visualization saved to: {output_path}")
    print(f"\nIf you're still having trouble seeing the point cloud:")
    print(f"1. Try different viewing angles by rotating the 3D view")
    print(f"2. Try a different browser if the current one isn't displaying properly")
    print(f"3. Check that the PLY file contains valid data points")

if __name__ == "__main__":
    main()
