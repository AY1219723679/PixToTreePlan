"""
Coordinate Utility Functions for PixToTreePlan

This module provides utilities for converting between different coordinate systems,
including 2D pixel coordinates and 3D point cloud coordinates.
"""

import numpy as np


def pixel_coords_to_3d(pixel_coords, depth_map, z_scale=1.0, normalize=False):
    """
    Convert pixel coordinates (x, y) to 3D coordinates (x, y, z) using a depth map.
    
    Args:
        pixel_coords (list or np.ndarray): List of (x, y) pixel coordinates 
            as tuples, lists, or a numpy array of shape (N, 2)
        depth_map (np.ndarray): 2D depth map of shape (height, width)
        z_scale (float, optional): Scale factor for the z-values. Defaults to 1.0.
        normalize (bool, optional): Whether to normalize the coordinates relative to 
            image dimensions. When True, coordinates are centered and normalized by 
            max(width, height). Defaults to False.
            
    Returns:
        np.ndarray: 3D coordinates as a numpy array of shape (N, 3) where each row is [x, y, z]
        
    Notes:
        - For normalized coordinates, (0, 0) is at the image center rather than top-left
        - The depth_map should be pre-processed (e.g., already normalized to 0-1 range if needed)
        - z_scale can be used to adjust the prominence of depth differences
    """
    # Convert input to numpy array if it's not already
    if not isinstance(pixel_coords, np.ndarray):
        pixel_coords = np.array(pixel_coords)
    
    # Ensure the pixel coordinates are integers (for indexing the depth map)
    pixel_coords_int = np.round(pixel_coords).astype(int)
    
    # Get height and width of depth map
    height, width = depth_map.shape
    
    # Create array for 3D coordinates
    coords_3d = np.zeros((len(pixel_coords), 3))
    
    # Filter out pixels that are outside the depth map bounds
    valid_indices = (
        (pixel_coords_int[:, 0] >= 0) & 
        (pixel_coords_int[:, 0] < width) & 
        (pixel_coords_int[:, 1] >= 0) & 
        (pixel_coords_int[:, 1] < height)
    )
    
    # Extract valid pixel coordinates and their indices in the original array
    valid_pixels = pixel_coords_int[valid_indices]
    valid_original_indices = np.where(valid_indices)[0]
    
    if normalize:
        # For normalized coordinates (centered at origin)
        normalization_factor = max(width, height)
        
        for i, (original_idx, (x, y)) in enumerate(zip(valid_original_indices, valid_pixels)):
            # Get depth value
            z = depth_map[y, x] * z_scale
            
            # Normalize coordinates
            norm_x = (x - width / 2) / normalization_factor
            norm_y = (height / 2 - y) / normalization_factor  # Flip Y axis
            
            # Store normalized coordinates
            coords_3d[original_idx] = [norm_x, norm_y, z]
    else:
        # For raw pixel coordinates (not normalized)
        for i, (original_idx, (x, y)) in enumerate(zip(valid_original_indices, valid_pixels)):
            # Get depth value
            z = depth_map[y, x] * z_scale
            
            # Store raw pixel coordinates with depth
            coords_3d[original_idx] = [x, y, z]
    
    return coords_3d


def pixel_coords_to_3d_batch(pixel_coords, depth_maps, z_scale=1.0, normalize=False):
    """
    Convert multiple sets of pixel coordinates to 3D coordinates using corresponding depth maps.
    
    Args:
        pixel_coords (list): List of arrays or lists, each containing (x, y) pixel coordinates
        depth_maps (list): List of depth maps corresponding to each set of pixel coordinates
        z_scale (float or list, optional): Scale factor(s) for the z-values. 
            If a list, should match length of pixel_coords. Defaults to 1.0.
        normalize (bool, optional): Whether to normalize the coordinates. Defaults to False.
            
    Returns:
        list: List of numpy arrays containing 3D coordinates
    """
    if not isinstance(z_scale, (list, tuple, np.ndarray)):
        z_scale = [z_scale] * len(pixel_coords)
        
    results = []
    for coords, depth_map, scale in zip(pixel_coords, depth_maps, z_scale):
        coords_3d = pixel_coords_to_3d(coords, depth_map, scale, normalize)
        results.append(coords_3d)
    
    return results


def load_depth_map(depth_map_path):
    """
    Load and normalize a depth map from file.
    
    Args:
        depth_map_path (str): Path to the depth map image file
        
    Returns:
        np.ndarray: Normalized depth map as a 2D numpy array
    """
    import cv2
    import os
    from PIL import Image
    import numpy as np
    
    # First check if file exists
    if not os.path.exists(depth_map_path):
        raise FileNotFoundError(f"Depth map file does not exist at {depth_map_path}")
        
    try:
        # Try using PIL first (better handling of Unicode paths)
        with Image.open(depth_map_path) as img:
            depth_map = np.array(img.convert('L'))
    except Exception as e:
        # Fall back to OpenCV if PIL fails
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
        
    if depth_map is None:
        raise FileNotFoundError(f"Could not load depth map from {depth_map_path}")
    
    # Normalize to 0-1 range
    depth_map = depth_map.astype(float) / 255.0
    
    return depth_map


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Generate a sample depth map (just for demonstration)
    sample_depth = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            # Create a simple gradient
            sample_depth[i, j] = (i + j) / 200
    
    # Create some test pixel coordinates
    test_coords = np.array([
        [20, 30],
        [50, 50],
        [80, 70],
        [40, 80]
    ])
    
    # Convert to 3D coordinates
    coords_3d_raw = pixel_coords_to_3d(test_coords, sample_depth)
    coords_3d_norm = pixel_coords_to_3d(test_coords, sample_depth, z_scale=0.5, normalize=True)
    
    print("Raw 3D coordinates:")
    print(coords_3d_raw)
    print("\nNormalized 3D coordinates:")
    print(coords_3d_norm)
    
    # Visualize the results
    fig = plt.figure(figsize=(12, 5))
    
    # Plot the depth map
    ax1 = fig.add_subplot(121)
    ax1.imshow(sample_depth, cmap='plasma')
    ax1.plot(test_coords[:, 0], test_coords[:, 1], 'ro', markersize=8)
    ax1.set_title('Depth Map with Test Points')
    
    # Plot the 3D points
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        coords_3d_norm[:, 0], 
        coords_3d_norm[:, 1], 
        coords_3d_norm[:, 2], 
        c='r', marker='o', s=100
    )
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z (Depth)')
    ax2.set_title('3D Visualization')
    
    plt.tight_layout()
    plt.savefig("coord_utils_test.png")
    print("Test visualization saved to coord_utils_test.png")
