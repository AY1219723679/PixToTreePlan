#!/usr/bin/env python3
"""
Point Cloud Static Views Generator

This script generates multiple static views of the point cloud from different angles.
"""

import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm

def main():
    # Path to the point cloud
    project_root = os.path.dirname(os.path.abspath(__file__))
    ply_path = os.path.join(project_root, "outputs", 
                           "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab", "point_cloud.ply")
    
    # Output directory for images
    output_dir = os.path.join(project_root, "point_cloud_views")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the point cloud
        print(f"Loading point cloud from: {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        # Create colors based on height if not available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            min_z = np.min(points[:, 2])
            max_z = np.max(points[:, 2])
            norm_z = (points[:, 2] - min_z) / (max_z - min_z)
            cmap = cm.get_cmap('viridis')
            colors = cmap(norm_z)[:, :3]
        
        print(f"Loaded {len(points)} points with coordinates range:")
        print(f"  X: [{np.min(points[:, 0]):.3f} to {np.max(points[:, 0]):.3f}]")
        print(f"  Y: [{np.min(points[:, 1]):.3f} to {np.max(points[:, 1]):.3f}]")
        print(f"  Z: [{np.min(points[:, 2]):.3f} to {np.max(points[:, 2]):.3f}]")
        
        # Generate views from different angles
        angles = [
            (0, 0),     # Top view
            (0, 90),    # Side view 1
            (90, 0),    # Side view 2
            (45, 45),   # Isometric view 1
            (45, 135),  # Isometric view 2
            (45, 225),  # Isometric view 3
            (45, 315)   # Isometric view 4
        ]
        
        print("\nGenerating views from different angles...")
        for i, (elev, azim) in enumerate(angles):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the points
            ax.scatter(
                points[:, 0], 
                points[:, 1], 
                points[:, 2], 
                c=colors,
                s=5.0,        # Larger point size for better visibility
                marker='.',
                alpha=1.0     # Full opacity
            )
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Point Cloud - View {i+1} (Elevation: {elev}°, Azimuth: {azim}°)')
            
            # Set the viewing angle
            ax.view_init(elev=elev, azim=azim)
            
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
            
            # Save the figure
            output_path = os.path.join(output_dir, f"view_{i+1}_elev{elev}_azim{azim}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"  Saved view {i+1} to {output_path}")
        
        print(f"\nGenerated {len(angles)} views of the point cloud in: {output_dir}")
        print("\nNow creating a composite image with multiple views...")
        
        # Create a composite image with all views
        fig, axs = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})
        axs = axs.flatten()
        
        for i, (elev, azim) in enumerate(angles):
            if i < len(axs):
                ax = axs[i]
                ax.scatter(
                    points[:, 0], 
                    points[:, 1], 
                    points[:, 2], 
                    c=colors,
                    s=1.0,
                    marker='.',
                    alpha=1.0
                )
                ax.view_init(elev=elev, azim=azim)
                ax.set_title(f'View {i+1}: Elev {elev}°, Azim {azim}°')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
        
        # Hide any unused subplots
        for i in range(len(angles), len(axs)):
            axs[i].axis('off')
        
        # Save the composite image
        composite_path = os.path.join(output_dir, "point_cloud_all_views.png")
        plt.tight_layout()
        plt.savefig(composite_path, dpi=300)
        print(f"Saved composite view to {composite_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
