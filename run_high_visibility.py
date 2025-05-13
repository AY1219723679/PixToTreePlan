#!/usr/bin/env python3
"""
Run the combined visualization with enhanced visibility

This script runs the combined visualization with larger point sizes
and higher opacity to ensure all elements are visible.
"""

import os
import sys
import subprocess

def main():
    # Set up paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Default test case
    ply_path = os.path.join(project_root, "outputs", 
                           "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab", "point_cloud.ply")
    
    if not os.path.exists(ply_path):
        print(f"Error: Default PLY file not found at {ply_path}")
        sys.exit(1)
    
    # Get related files
    image_path = os.path.join(project_root, "outputs", 
                             "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab", "original.png")
    depth_path = os.path.join(project_root, "outputs", 
                             "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab", "depth_map.png")
    label_path = os.path.join(project_root, "YOLO", "train", "labels", 
                             "urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.txt")
    
    # Output path
    output_path = os.path.join(project_root, "high_visibility_viz.html")
      # Build command with MUCH larger points for maximum visibility
    cmd = [
        sys.executable,
        os.path.join(project_root, "visualize_combined_pointclouds.py"),
        f"--ply={ply_path}",
        f"--image={image_path}",
        f"--depth={depth_path}",
        f"--label={label_path}",
        f"--output={output_path}",
        "--point_size=15.0",   # MUCH larger point size for maximum visibility
        "--center_size=20.0",  # Very large center points 
        "--title=EXTRA HIGH Visibility 3D Visualization"
    ]
    
    # Run visualization
    print("\nRunning high visibility visualization...")
    subprocess.run(cmd)
    
    print(f"\nVisualization saved to: {output_path}")
    print("\nViewing Tips:")
    print("1. Use left-click + drag to rotate the 3D view")
    print("2. Use right-click + drag to pan")
    print("3. Use scroll wheel to zoom in/out")
    print("4. Try rotating the view to see the point cloud from different angles")

if __name__ == "__main__":
    main()
