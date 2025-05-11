"""
Convert Image to Point Cloud with Open3D

This script demonstrates how to generate a 3D point cloud from a 2D cutout image using Open3D.  
We assume the image (`cutout_ground.png`) contains an isolated object or surface with a transparent or masked background.

Dependencies:
- Open3D
- NumPy
- PIL (Pillow)
- PyTorch and torchvision (for MiDaS depth estimation)

Steps:
1. Load image and convert to grayscale or extract non-transparent regions.
2. Assign pixel locations as (x, y) coordinates.
3. Use MiDaS to estimate depth for z values.
4. Generate point cloud and visualize it in 3D.

Reference:
Based on Open3D's official geometry tutorial:
https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
"""

import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import cv2
import urllib.request

def load_midas_model():
    """
    Load the MiDaS model for depth estimation.
    
    Returns:
        model: The MiDaS depth estimation model
        transform: The preprocessing transformation for the model
    """
    # Check if model is already downloaded
    midas_model_path = "midas_model"
    os.makedirs(midas_model_path, exist_ok=True)
    model_path = os.path.join(midas_model_path, "model-small.pt")
    
    # Download the model if not available
    if not os.path.exists(model_path):
        print("Downloading MiDaS model...")
        url = "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt"
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded to {model_path}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Suppress warnings by specifying trust_repo=True
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to(device)
    model.eval()
    
    # Load transformation
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    
    return model, transform, device

def visualize_depth_map(depth_map, save_path=None):
    """
    Visualize a depth map as a heatmap.
    
    Args:
        depth_map (np.ndarray): The depth map to visualize
        save_path (str): Optional path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth')
    plt.title('MiDaS Depth Map')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Depth map visualization saved to {save_path}")
    
    plt.close()

def image_to_pointcloud(image_path, use_alpha=True, z_scale=0.1, sample_rate=1, save_depth_map=False):
    """
    Convert an image to a point cloud.
    
    Args:
        image_path (str): Path to the image
        use_alpha (bool): If True, only use pixels where alpha > 0
        z_scale (float): Scale factor for z-values (depth)
        sample_rate (int): Sample every nth pixel to reduce point cloud density
        save_depth_map (bool): Whether to save the depth map visualization
        
    Returns:
        o3d.geometry.PointCloud: The generated point cloud
    """
    print(f"Loading image from {image_path}")
    # Load the image with Pillow
    img = Image.open(image_path)
    img_np = np.array(img)
    
    # Get image dimensions
    height, width = img_np.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Load MiDaS model for depth estimation
    print("Loading MiDaS depth estimation model...")
    model, transform, device = load_midas_model()
    
    # Prepare input for MiDaS
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Apply input transforms
    input_batch = transform(input_image).to(device)
    
    # Predict and resize depth
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Get depth map as numpy array
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth values to 0-1 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    
    # Optionally save depth map visualization
    if save_depth_map:
        # Create file path for depth map visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        depth_dir = "output_depth"
        os.makedirs(depth_dir, exist_ok=True)
        depth_vis_path = os.path.join(depth_dir, f"{base_name}_depth_midas.png")
        
        # Save visualization
        visualize_depth_map(depth_map, depth_vis_path)
    
    # Create lists for points and colors
    points = []
    colors = []
    
    # Process the image
    for y in range(0, height, sample_rate):
        for x in range(0, width, sample_rate):
            # Check if this pixel should be included
            if use_alpha and img_np.shape[2] == 4:  # RGBA image
                if img_np[y, x, 3] == 0:  # Skip fully transparent pixels
                    continue
            
            # Get RGB values (normalize to 0-1)
            if img_np.shape[2] >= 3:  # RGB or RGBA
                color = img_np[y, x, :3].astype(float) / 255.0
            else:  # Grayscale
                gray_value = float(img_np[y, x]) / 255.0
                color = np.array([gray_value, gray_value, gray_value])
            
            # Get Z value from MiDaS depth map
            z = depth_map[y, x] * z_scale
            
            # Normalize coordinates to center the point cloud
            # X and Y are flipped and centered, Z points upward
            norm_x = (x - width / 2) / max(width, height)
            norm_y = (height / 2 - y) / max(width, height)
            
            # Add the point and its color
            points.append([norm_x, norm_y, z])
            colors.append(color)
    
    print(f"Generated {len(points)} points")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    return pcd

def visualize_pointcloud(pcd, save_path=None):
    """
    Visualize the point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to visualize
        save_path (str): Optional path to save the visualization
    """
    # Create visualization
    print("Visualizing point cloud...")
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                     window_name="Image to Point Cloud",
                                     width=1024, height=768,
                                     point_show_normal=False)
    
    if save_path:
        print(f"Saving visualization to {save_path}")
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1024, height=768)
        vis.add_geometry(pcd)
        vis.add_geometry(coordinate_frame)
        vis.run()
        vis.capture_screen_image(save_path)
        vis.destroy_window()

def save_pointcloud(pcd, output_path):
    """
    Save the point cloud to a file.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to save
        output_path (str): Path to save the point cloud file
    """
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")

if __name__ == "__main__":
    # Input and output paths
    input_image = "cutout_ground.png"
    output_dir = "output_pointcloud"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    output_pointcloud = os.path.join(output_dir, "ground_pointcloud.ply")
    output_visualization = os.path.join(output_dir, "pointcloud_visualization.png")
    
    print("Converting image to point cloud...")
    
    # Convert image to point cloud
    # You can adjust parameters for different results:
    # - Lower sample_rate for higher density
    # - Higher z_scale for more pronounced elevation
    # - Set save_depth_map=True to visualize the depth map
    pcd = image_to_pointcloud(input_image, use_alpha=True, z_scale=1, sample_rate=2, save_depth_map=True)
    
    # Optional: Apply additional processing
    print("Applying statistical outlier removal...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Optional: Estimate normals for better visualization
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location()
    
    # Save the point cloud
    save_pointcloud(pcd, output_pointcloud)
    
    # Visualize the point cloud
    visualize_pointcloud(pcd, output_visualization)
    
    print("Process completed!")
