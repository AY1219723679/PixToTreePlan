#!/usr/bin/env python3
"""
Direct Open3D Visualization

This script uses Open3D's built-in visualizer to display the point cloud.
"""

import os
import open3d as o3d

# Path to point cloud
ply_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       "outputs", "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab", "point_cloud.ply")

print(f"Loading point cloud from: {ply_path}")
try:
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Print basic info
    print(f"Point cloud contains {len(pcd.points)} points")
    
    # Add a coordinate system for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Create visualization directly using Open3D's built-in visualizer
    print("Launching Open3D visualizer...")
    print("Close the visualizer window to continue")
    o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                     window_name="Direct Point Cloud Visualization",
                                     width=1024, height=768)
    
except Exception as e:
    print(f"Error: {e}")
