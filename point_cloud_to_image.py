#!/usr/bin/env python3
"""
Point Cloud Visualization to Image

This script visualizes the point cloud and saves it as an image file.
"""

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm

def main():
    # Path to point cloud
    ply_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           "outputs", "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab", "point_cloud.ply")
    
    # Output path for image
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "point_cloud_visualization.png")
    
    print(f"Loading point cloud from: {ply_path}")
    
    try:
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        # Get colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            # Create a color gradient based on height (Z)
            min_z = np.min(points[:, 2])
            max_z = np.max(points[:, 2])
            norm_z = (points[:, 2] - min_z) / (max_z - min_z)
            
            # Use a colormap
            cmap = cm.get_cmap('viridis')
            colors = cmap(norm_z)[:, :3]  # Drop alpha channel
        
        # Print point cloud statistics
        print(f"Loaded {len(points)} points")
        print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        
        # Create a figure for 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the point cloud
        ax.scatter(
            points[:, 0],  # x
            points[:, 1],  # y
            points[:, 2],  # z
            c=colors,      # colors
            s=1.0,         # point size
            marker='.',    # marker style
            alpha=0.8      # transparency
        )
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Visualization')
        
        # Set equal aspect ratio
        max_range = np.max([
            np.max(points[:, 0]) - np.min(points[:, 0]),
            np.max(points[:, 1]) - np.min(points[:, 1]),
            np.max(points[:, 2]) - np.min(points[:, 2])
        ])
        mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2
        mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
        mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) / 2
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Set a good viewing angle
        ax.view_init(elev=30, azim=45)
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        
        # Show the figure
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
