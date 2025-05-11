import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import shutil
import importlib.util

# Add project modules to path - using absolute paths to ensure they work correctly
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_dir, "main", "get_ground_mask"))
sys.path.append(os.path.join(curr_dir, "main", "generate_depth"))
sys.path.append(os.path.join(curr_dir, "main", "img_to_pointcloud"))

# Import the modules with explicit error handling for each import
try:
    from main.get_ground_mask.single_image_processing import get_model, process_image, extract_ground_mask
    print("Successfully imported ground mask modules")
except ImportError as e:
    print(f"Error importing ground mask modules: {e}")
    print("Please make sure the ground mask modules are installed correctly")
    sys.exit(1)

try:
    from main.get_ground_mask.create_cutout_simple import create_cutout_with_mask
    print("Successfully imported cutout module")
except ImportError as e:
    print(f"Error importing cutout module: {e}")
    print("Please make sure the cutout module is installed correctly")
    sys.exit(1)

try:
    from main.generate_depth.simple_depth_generator import generate_depth_map
    print("Successfully imported depth map module")
except ImportError as e:
    print(f"Error importing depth map module: {e}")
    print("Please make sure the depth map module is installed correctly")
    sys.exit(1)

# Only import point cloud modules if Open3D is available
POINT_CLOUD_AVAILABLE = False
try:
    import open3d as o3d
    from main.img_to_pointcloud.image_to_pointcloud import image_to_pointcloud, visualize_pointcloud, save_pointcloud
    POINT_CLOUD_AVAILABLE = True
    print("Successfully imported point cloud modules")
except ImportError as e:
    print(f"Warning: {e}")
    print("Point cloud generation will be disabled")
    print("To enable point cloud generation, install Open3D with: pip install open3d")

def setup_directories():
    """
    Create necessary output directories if they don't exist
    """
    os.makedirs("output_depth", exist_ok=True)
    os.makedirs("output_groundmasks", exist_ok=True)
    os.makedirs("output_pointcloud", exist_ok=True)

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='PixToTreePlan: Process an image through the entire pipeline')
    
    parser.add_argument('--image_path', type=str, default=None, 
                        help='Path to the input image')
    parser.add_argument('--resize_factor', type=float, default=0.6,
                        help='Resize factor for segmentation (lower = coarser, range: 0.3-1.0)')
    parser.add_argument('--min_region_size', type=int, default=400,
                        help='Minimum size of regions to preserve (in pixels)')
    parser.add_argument('--min_area_ratio', type=float, default=0.1, 
                        help='Min component size as percentage of image area (0.1 = 10%)')
    parser.add_argument('--z_scale', type=float, default=0.5,
                        help='Scale factor for Z values in point cloud')
    parser.add_argument('--sample_rate', type=int, default=2,
                        help='Sample rate for point cloud generation (1=full density)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations for each step')
    
    return parser.parse_args()

def step1_generate_groundmask(image_path, args):
    """
    Step 1: Generate a ground mask using semantic segmentation
    """
    print("\nSTEP 1: Generating ground mask...")
    
    # Get model
    model = get_model()
    
    # Process image
    img, segmentation_map = process_image(
        model, 
        image_path, 
        resize_factor=args.resize_factor,
        smooth_output=True,
        min_region_size=args.min_region_size
    )
    
    # Extract ground mask
    ground_mask = extract_ground_mask(
        segmentation_map, 
        min_area_ratio=args.min_area_ratio
    )
    
    # Get base filename without extension
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the mask
    mask_path = os.path.join("ground_mask.png")
    Image.fromarray((ground_mask * 255).astype(np.uint8)).save(mask_path)
    
    # Also save a copy in output_groundmasks with the original image name
    output_mask_path = os.path.join("output_groundmasks", f"{image_basename}_groundmask.png")
    Image.fromarray((ground_mask * 255).astype(np.uint8)).save(output_mask_path)
    
    print(f"  Ground mask saved to {mask_path}")
    print(f"  Copy saved to {output_mask_path}")
    
    return mask_path, image_basename

def step2_create_cutout(image_path, mask_path, image_basename):
    """
    Step 2: Create a cutout image using the ground mask
    """
    print("\nSTEP 2: Creating cutout image...")
    
    # Ensure image_path is an absolute path
    image_path = os.path.abspath(image_path)
    print(f"  Using absolute image path: {image_path}")
    
    # Create cutout
    cutout_path = "cutout_ground.png"
    create_cutout_with_mask(image_path, mask_path, cutout_path)
    
    # Also save a copy in output_groundmasks with the original image name
    output_cutout_path = os.path.join("output_groundmasks", f"{image_basename}_cutout.png")
    shutil.copy(cutout_path, output_cutout_path)
    
    print(f"  Cutout saved to {cutout_path}")
    print(f"  Copy saved to {output_cutout_path}")
    
    return cutout_path

def step3_generate_depth_map(cutout_path):
    """
    Step 3: Generate a depth map using MiDaS
    """
    print("\nSTEP 3: Generating depth map...")
    
    # Generate depth map
    output_dir = "output_depth"
    generate_depth_map(cutout_path=cutout_path, output_dir=output_dir)
    
    # The depth map is saved by the function
    depth_map_path = os.path.join(output_dir, "depth_masked.png")
    print(f"  Depth map saved to {depth_map_path}")
    
    return depth_map_path

def step4_create_pointcloud(cutout_path, image_basename, args):
    """
    Step 4: Create a point cloud from the cutout image and depth map
    """
    print("\nSTEP 4: Generating point cloud...")
    
    # Import Open3D here to ensure it's available
    import open3d as o3d
    
    # Define output paths
    output_pointcloud = os.path.join("output_pointcloud", f"{image_basename}_pointcloud.ply")
    output_visualization = os.path.join("output_pointcloud", f"{image_basename}_visualization.png")
    
    # Generate point cloud
    pcd = image_to_pointcloud(
        cutout_path, 
        use_alpha=True, 
        z_scale=args.z_scale, 
        sample_rate=args.sample_rate, 
        save_depth_map=True
    )
    
    # Optional: Apply statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Estimate normals for better visualization
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location()
    
    # Save the point cloud
    save_pointcloud(pcd, output_pointcloud)
    
    # Visualize the point cloud if requested
    if args.visualize:
        visualize_pointcloud(pcd, output_visualization)
    
    print(f"  Point cloud saved to {output_pointcloud}")
    if args.visualize:
        print(f"  Visualization saved to {output_visualization}")
    
    return output_pointcloud

def main():
    print("======================================================")
    print("PixToTreePlan: Image to Point Cloud Processing Pipeline")
    print("======================================================")
    
    # Set up directories
    setup_directories()
    
    # Parse command line arguments
    args = parse_arguments()
      # Default image if none provided
    if args.image_path is None:
        # Check both possible input image paths
        default_image = "urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg"
        std_path = os.path.join("input_images", default_image)
        dataset_path = os.path.join("dataset", "input_images", default_image)
        
        if os.path.exists(std_path):
            args.image_path = std_path
        elif os.path.exists(dataset_path):
            args.image_path = dataset_path
        else:
            # Just use the standard path even if it doesn't exist
            args.image_path = std_path
            print(f"Warning: Default image not found. Using path: {args.image_path}")
            
        print(f"No image specified, using default: {args.image_path}")
    
    # Display parameters
    print("\nProcessing Parameters:")
    print(f"  - Image path: {args.image_path}")
    print(f"  - Resize factor: {args.resize_factor}")
    print(f"  - Min region size: {args.min_region_size} pixels")
    print(f"  - Min area ratio: {args.min_area_ratio * 100}%")
    print(f"  - Z scale: {args.z_scale}")
    print(f"  - Sample rate: {args.sample_rate}")
    print(f"  - Visualize: {args.visualize}")
    
    try:
        # Step 1: Generate ground mask
        mask_path, image_basename = step1_generate_groundmask(args.image_path, args)
          # Step 2: Create cutout image
        cutout_path = step2_create_cutout(args.image_path, mask_path, image_basename)
        
        # Step 3: Generate depth map
        depth_map_path = step3_generate_depth_map(cutout_path)
        
        # Step 4: Create point cloud
        if 'POINT_CLOUD_AVAILABLE' in globals() and POINT_CLOUD_AVAILABLE:
            try:
                output_pointcloud = step4_create_pointcloud(cutout_path, image_basename, args)
                print("\nPoint cloud generation completed successfully!")
            except Exception as e:
                print(f"\nERROR in point cloud generation: {str(e)}")
                print("Point cloud generation skipped.")
        else:
            print("\nWARNING: Open3D not found. Point cloud generation skipped.")
            print("To generate point clouds, install Open3D with: pip install open3d")
        
        print("\n======================================================")
        print("Processing Complete!")
        print(f"Ground Mask: {mask_path}")
        print(f"Cutout Image: {cutout_path}")
        print(f"Depth Map: {depth_map_path}")
        if 'output_pointcloud' in locals():
            print(f"Point Cloud: {output_pointcloud}")
        print("======================================================")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
