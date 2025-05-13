"""
Convert Image to Point Cloud with Open3D

This script demonstrates how to generate a 3D point cloud from a 2D cutout image using Open3D.  
We assume the image (`cutout_ground.png`) contains an isolated object or surface with a transparent or masked background.

Dependencies:
- Open3D
- NumPy
- PIL (Pillow)
- PyTorch and torchvision (for MiDaS depth estimation)
- Matplotlib (for 3D visualization)

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
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
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

def image_to_pointcloud(image_path, use_alpha=True, z_scale=0.1, sample_rate=1, save_depth_map=False, use_normalized_coords=True):
    """
    Convert an image to a point cloud.
    
    Args:
        image_path (str): Path to the image
        use_alpha (bool): If True, only use pixels where alpha > 0
        z_scale (float): Scale factor for z-values (depth)
        sample_rate (int): Sample every nth pixel to reduce point cloud density
        save_depth_map (bool): Whether to save the depth map visualization
        use_normalized_coords (bool): If True, normalize coordinates for Open3D visualization
                                      If False, use original pixel coordinates
        
    Returns:
        tuple: (pcd, raw_points, colors) where:
               - pcd is the normalized Open3D point cloud (o3d.geometry.PointCloud)
               - raw_points is a numpy array of (x, y, z) with original pixel coordinates
               - colors is a numpy array of RGB colors for each point
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
    pil_image = Image.open(image_path)
    input_image = np.array(pil_image)
    # If image has alpha channel, take only RGB components
    if len(input_image.shape) > 2 and input_image.shape[2] == 4:  # RGBA
        input_image = input_image[:, :, :3]  # Just take RGB components
    # PIL already gives RGB format, no need to convert
    
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
      # Depth map visualization is now handled by the main script
    # This option is kept for backward compatibility but doesn't save files
    if save_depth_map:
        print("Note: Depth maps are now saved in the image output directory instead of separate folders")
    
    # Create lists for points and colors (normalized for Open3D)
    normalized_points = []
    # Create lists for raw points with original pixel coordinates (for matplotlib)
    raw_points = []
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
            
            # Store the original pixel coordinates for matplotlib visualization
            raw_points.append([x, y, z])
            colors.append(color)
            
            # Normalize coordinates to center the point cloud for Open3D
            # X and Y are flipped and centered, Z points upward
            norm_x = (x - width / 2) / max(width, height)
            norm_y = (height / 2 - y) / max(width, height)
            normalized_points.append([norm_x, norm_y, z])
    
    print(f"Generated {len(raw_points)} points")
    
    # Convert to numpy arrays
    raw_points = np.array(raw_points)
    colors = np.array(colors)
    normalized_points = np.array(normalized_points)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(normalized_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd, raw_points, colors

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

def visualize_pointcloud_matplotlib(points, colors, save_path=None):
    """
    Visualize the point cloud using matplotlib's 3D scatter plot.
    The plot will use original image pixel coordinates for X and Y axes.
    
    Args:
        points (np.ndarray): Array of points (x, y, z) where x and y are image coordinates 
                            and z is depth
        colors (np.ndarray): Array of RGB colors for each point
        save_path (str): Optional path to save the visualization
    """
    # Create a new figure with 3D projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    
    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Scatter plot with RGB colors
    scatter = ax.scatter(x, y, z, c=colors, s=1)
    
    # Set labels and title
    ax.set_xlabel('X (Image Width)')
    ax.set_ylabel('Y (Image Height)')
    ax.set_zlabel('Z (Depth)')
    ax.set_title('Ground Depth Point Cloud')
    
    # Add a color bar
    plt.colorbar(scatter, ax=ax, label='RGB Color')
    
    # Adjust view angle for better initial perspective
    ax.view_init(elev=30, azim=45)
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Matplotlib 3D visualization saved to {save_path}")
    
    # Show the plot (interactive)
    plt.tight_layout()
    plt.show()

def generate_mesh_from_depth_and_cutout(image_path, z_scale=1.0, save_path=None, poisson_depth=9):
    """
    Generate a 3D mesh from an image using MiDaS-predicted depth.
    
    Args:
        image_path (str): Path to the cutout image
        z_scale (float): Scale factor for depth values
        save_path (str): Optional path to save the generated mesh (.ply or .obj)
        poisson_depth (int): Depth parameter for Poisson surface reconstruction
        
    Returns:
        o3d.geometry.TriangleMesh: The generated mesh
    """
    print(f"Generating 3D mesh from {image_path}")
    
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
    pil_image = Image.open(image_path)
    input_image = np.array(pil_image)
    # If image has alpha channel, take only RGB components
    if len(input_image.shape) > 2 and input_image.shape[2] == 4:  # RGBA
        input_image = input_image[:, :, :3]  # Just take RGB components
    # PIL already gives RGB format, no need to convert
    
    # Apply input transforms
    input_batch = transform(input_image).to(device)
    
    # Predict and resize depth
    print("Predicting depth...")
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
    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    
    # Apply z_scale to depth values
    depth_map_scaled = depth_map_normalized * z_scale
    
    # If image has alpha channel, use it as mask
    if img_np.shape[2] == 4:
        print("Using alpha channel as mask...")
        # Set depth to 0 where alpha is 0
        mask = img_np[:, :, 3] == 0
        depth_map_scaled[mask] = 0
    
    # Convert RGB and depth to Open3D images
    print("Converting to Open3D images...")
    # Create color image using RGB channels
    rgb_image = o3d.geometry.Image(img_np[:, :, :3].astype(np.uint8))
    
    # Create depth image
    depth_image = o3d.geometry.Image(depth_map_scaled.astype(np.float32))
    
    # Create RGBD image from color and depth
    print("Creating RGBD image...")
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image,
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    
    # Create camera intrinsic parameters
    print("Creating camera intrinsics...")
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=width,
        height=height,
        fx=width,  # Using width as focal length for better scaling
        fy=width,  # Same as fx for square pixels
        cx=width / 2,  # Principal point at center
        cy=height / 2
    )
    
    # Create point cloud from RGBD image
    print("Creating point cloud from RGBD image...")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    
    # Flip the point cloud to correct orientation if needed
    print("Adjusting point cloud orientation...")
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # Optional: Apply statistical outlier removal
    print("Applying statistical outlier removal...")
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Estimate normals for mesh generation
    print("Estimating normals...")
    pcd_filtered.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd_filtered.orient_normals_towards_camera_location()
    
    # Create a mesh using Poisson surface reconstruction
    print(f"Generating mesh with Poisson reconstruction (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_filtered, depth=poisson_depth
    )
    
    # Optional: Remove low density vertices
    print("Removing low-density vertices...")
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Save mesh if requested
    if save_path:
        print(f"Saving mesh to {save_path}")
        o3d.io.write_triangle_mesh(save_path, mesh)
    
    return mesh

def visualize_mesh(mesh, save_path=None):
    """
    Visualize a 3D mesh using Open3D.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to visualize
        save_path (str): Optional path to save the visualization
    """
    # Create visualization
    print("Visualizing mesh...")
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    
    # Visualize
    o3d.visualization.draw_geometries(
        [mesh, coordinate_frame], 
        window_name="3D Mesh from Depth",
        width=1024, height=768,
        mesh_show_wireframe=True
    )
    
    if save_path:
        print(f"Saving mesh visualization to {save_path}")
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1024, height=768)
        vis.add_geometry(mesh)
        vis.add_geometry(coordinate_frame)
        vis.run()
        vis.capture_screen_image(save_path)
        vis.destroy_window()

def visualize_mesh_matplotlib(mesh, save_path=None):
    """
    Visualize a 3D mesh using Matplotlib.
    This provides an alternative visualization to Open3D.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to visualize
        save_path (str): Optional path to save the visualization
    """
    print("Visualizing mesh with matplotlib...")
    
    # Get vertex and triangle arrays from the mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the 3D plot - we'll use Poly3DCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Extract triangles as a list of vertex triplets
    triangle_vertices = vertices[triangles]
    
    # Create the collection with colors
    mesh_collection = Poly3DCollection(
        triangle_vertices,
        alpha=0.7,
        edgecolor='k',
        linewidth=0.2
    )
    
    # Add color based on vertex colors or depth gradient
    if len(vertex_colors) > 0:
        triangle_colors = np.mean(vertex_colors[triangles], axis=1)
        mesh_collection.set_facecolor(triangle_colors)
    else:
        z_vals = np.mean(triangle_vertices[:, :, 2], axis=1)
        norm = plt.Normalize(z_vals.min(), z_vals.max())
        colors = plt.cm.viridis(norm(z_vals))
        mesh_collection.set_facecolor(colors)
    
    # Add collection to axis, set bounds, labels, title
    ax.add_collection3d(mesh_collection)
    ax.set_xlim(vertices.min(axis=0)[0], vertices.max(axis=0)[0])
    ax.set_ylim(vertices.min(axis=0)[1], vertices.max(axis=0)[1])
    ax.set_zlim(vertices.min(axis=0)[2], vertices.max(axis=0)[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mesh Visualization (Matplotlib)')
    
    # Add colorbar, set view, save, and show
    if len(vertex_colors) == 0:
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        scalar_map.set_array([])
        plt.colorbar(scalar_map, ax=ax, label='Z Depth')
    
    ax.view_init(elev=30, azim=45)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Matplotlib mesh visualization saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert images to 3D point clouds or meshes")
    parser.add_argument("--image_path", default="outputs/park-trees_19_jpg_rf_8d6908c164aaf4b773199fbc190dc552/cutout.png",
                        help="Path to input image")
    parser.add_argument("--output_dir", default=os.path.join("outputs", "debug_pointcloud"),
                        help="Directory for output files")
    parser.add_argument("--mode", choices=["pointcloud", "mesh", "both"], default="both",
                        help="Processing mode: 'pointcloud', 'mesh', or 'both'")
    parser.add_argument("--z_scale", type=float, default=1.0,
                        help="Scale factor for depth values")
    parser.add_argument("--sample_rate", type=int, default=2,
                        help="Sample every nth pixel for point cloud generation")
    parser.add_argument("--poisson_depth", type=int, default=9,
                        help="Depth parameter for Poisson surface reconstruction (higher values = more detail)")
    parser.add_argument("--save_depth_map", action="store_true",
                        help="Save depth map visualization")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output file paths
    output_pointcloud = os.path.join(args.output_dir, "ground_pointcloud.ply")
    output_visualization = os.path.join(args.output_dir, "pointcloud_visualization.png")
    output_matplotlib = os.path.join(args.output_dir, "pointcloud_matplotlib_3d.png")
    output_mesh = os.path.join(args.output_dir, "ground_mesh.ply")
    output_mesh_visualization = os.path.join(args.output_dir, "mesh_visualization.png")
    
    # Process based on the selected mode
    if args.mode in ["pointcloud", "both"]:
        print("Converting image to point cloud...")
        
        # Convert image to point cloud
        pcd, raw_points, colors = image_to_pointcloud(
            args.image_path, 
            use_alpha=True, 
            z_scale=args.z_scale, 
            sample_rate=args.sample_rate, 
            save_depth_map=args.save_depth_map
        )
        
        # Optional: Apply additional processing
        print("Applying statistical outlier removal...")
        pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Optional: Estimate normals for better visualization
        print("Estimating normals...")
        pcd_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_filtered.orient_normals_towards_camera_location()
        
        # Save the point cloud
        save_pointcloud(pcd_filtered, output_pointcloud)
        
        # Visualize with Open3D
        print("Generating Open3D visualization...")
        visualize_pointcloud(pcd_filtered, output_visualization)
          # Visualize with matplotlib 3D (using original pixel coordinates)
        print("Generating matplotlib 3D visualization...")
        visualize_pointcloud_matplotlib(raw_points, colors, output_matplotlib)
    
    if args.mode in ["mesh", "both"]:
        print("Generating 3D mesh...")
        
        # Generate mesh from depth and image
        mesh = generate_mesh_from_depth_and_cutout(
            args.image_path,
            z_scale=args.z_scale,
            save_path=output_mesh,
            poisson_depth=args.poisson_depth
        )
        
        # Visualize the mesh with Open3D
        print("Visualizing 3D mesh with Open3D...")
        visualize_mesh(mesh, save_path=output_mesh_visualization)
        
        # Visualize the mesh with Matplotlib
        print("Visualizing 3D mesh with Matplotlib...")
        output_mesh_matplotlib = os.path.join(args.output_dir, "mesh_matplotlib_3d.png")
        visualize_mesh_matplotlib(mesh, save_path=output_mesh_matplotlib)
    
    print("Process completed!")
