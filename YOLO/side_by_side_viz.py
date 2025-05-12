"""
Side-by-side point cloud visualization for YOLO to 3D demo
"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_point_clouds_side_by_side(coords_3d_1, coords_3d_2=None, 
                                       title1="3D Points from Original Depth Map", 
                                       title2="3D Points from Ground-Only Depth Map"):
    """
    Create a side-by-side visualization of two point clouds for comparison.
    
    Args:
        coords_3d_1 (np.ndarray): First 3D point cloud coordinates
        coords_3d_2 (np.ndarray, optional): Second 3D point cloud coordinates
        title1 (str): Title for the first point cloud
        title2 (str): Title for the second point cloud
    """
    # Create a figure for side-by-side visualization
    comparison_fig = plt.figure(figsize=(20, 10))
    
    # Determine common axis limits for consistent visualization
    x_min = min(coords_3d_1[:, 0].min(), coords_3d_2[:, 0].min())
    x_max = max(coords_3d_1[:, 0].max(), coords_3d_2[:, 0].max())
    y_min = min(coords_3d_1[:, 1].min(), coords_3d_2[:, 1].min())
    y_max = max(coords_3d_1[:, 1].max(), coords_3d_2[:, 1].max())
    z_min = min(coords_3d_1[:, 2].min(), coords_3d_2[:, 2].min())
    z_max = max(coords_3d_1[:, 2].max(), coords_3d_2[:, 2].max())
    
    # Create common normalization for coloring
    norm = plt.Normalize(z_min, z_max)
    
    # First subplot - original point cloud
    ax1 = comparison_fig.add_subplot(121, projection='3d')
    colors1 = plt.cm.plasma(norm(coords_3d_1[:, 2]))
    ax1.scatter(
        coords_3d_1[:, 0],
        coords_3d_1[:, 1],
        coords_3d_1[:, 2],
        c=colors1, 
        s=30,
        alpha=0.8
    )
    
    # Set consistent axes limits
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.set_zlim([z_min, z_max])
    
    # Labels and title
    ax1.set_xlabel('X (pixel)')
    ax1.set_ylabel('Y (pixel)')
    ax1.set_zlabel('Z (depth)')
    ax1.set_title(title1)
    
    # Second subplot - masked point cloud
    ax2 = comparison_fig.add_subplot(122, projection='3d')
    colors2 = plt.cm.plasma(norm(coords_3d_2[:, 2]))
    ax2.scatter(
        coords_3d_2[:, 0],
        coords_3d_2[:, 1],
        coords_3d_2[:, 2],
        c=colors2, 
        s=30,
        alpha=0.8
    )
    
    # Set consistent axes limits
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_zlim([z_min, z_max])
    
    # Labels and title
    ax2.set_xlabel('X (pixel)')
    ax2.set_ylabel('Y (pixel)')
    ax2.set_zlabel('Z (depth)')
    ax2.set_title(title2)
    
    # Add a color bar to show the depth scale
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap='plasma')
    scalar_map.set_array([])
    cbar = comparison_fig.colorbar(scalar_map, ax=[ax1, ax2], label='Depth')
    
    # Set the same view angle for both subplots
    ax1.view_init(elev=30, azim=45)
    ax2.view_init(elev=30, azim=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the figure in a non-blocking way
    plt.show(block=False)
    
    return comparison_fig
