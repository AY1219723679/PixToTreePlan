#!/usr/bin/env python3
"""
Batch Image Processing Script for PixToTreePlan

This script processes all images in a directory through the PixToTreePlan pipeline.
It generates ground masks, depth maps, and point clouds for each image.
"""

import os
import sys
import argparse
import glob
import time
from tqdm import tqdm

# Check if Open3D is available for point cloud generation
POINT_CLOUD_AVAILABLE = False
try:
    import open3d as o3d
    POINT_CLOUD_AVAILABLE = True
except ImportError:
    pass

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process multiple images through the PixToTreePlan pipeline')
    
    parser.add_argument('--input_dir', type=str, default='input_images',
                        help='Directory containing input images')
    parser.add_argument('--extensions', type=str, default='jpg,jpeg,png',
                        help='File extensions to process (comma-separated)')
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
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    
    return parser.parse_args()

def process_image(image_path, args):
    """Process a single image through the pipeline"""
    print(f"\nProcessing image: {image_path}")
    
    # Get the absolute paths
    core_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(core_dir)
    image_path = os.path.abspath(image_path)
    
    # Get image basename for output
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Image basename: {image_basename}")
    
    # Build command to process this image
    cmd = [
        sys.executable,
        os.path.join(core_dir, 'main.py'),
        f'--image_path={image_path}',
        f'--resize_factor={args.resize_factor}',
        f'--min_region_size={args.min_region_size}',
        f'--min_area_ratio={args.min_area_ratio}',
        f'--z_scale={args.z_scale}',
        f'--sample_rate={args.sample_rate}'
    ]
    
    if args.visualize:
        cmd.append('--visualize')
    
    # Call the main script
    import subprocess
    result = subprocess.run(cmd, capture_output=False, text=True)
      # Show error if any occurred
    if result.stderr:
        print(f"Error processing {image_path}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
      # Check if output directory exists
    # Ensure we use the project root for outputs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", image_basename.replace('.', '_').replace(' ', '_'))
    
    if os.path.isdir(output_dir):
        print(f"Output saved to: {output_dir}")
        # List files in output directory
        files = os.listdir(output_dir)
        
        # Check for key output files
        has_depth = any("depth" in f for f in files)
        has_pc = any("point_cloud" in f for f in files)
        
        success = has_depth and (not POINT_CLOUD_AVAILABLE or has_pc)
    else:
        success = False
        
    return success

def find_input_directory(input_dir, project_root):
    """
    Find the correct input directory by checking multiple possible locations
    
    Args:
        input_dir: The input directory path provided by the user
        project_root: The root directory of the project
    """
    # If input_dir is an absolute path, use it directly
    if os.path.isabs(input_dir) and os.path.isdir(input_dir):
        if has_image_files(input_dir):
            print(f"Using provided absolute input directory: {input_dir}")
            return input_dir
    
    # Check if the relative path from current directory exists
    if os.path.isdir(input_dir):
        if has_image_files(input_dir):
            print(f"Using provided input directory: {input_dir}")
            return input_dir
    
    # Try other common paths relative to project root
    possible_paths = [
        input_dir,
        os.path.join(project_root, input_dir),
        os.path.join(project_root, "input_images"),
        os.path.join(project_root, "dataset", input_dir),
        os.path.join(project_root, "dataset", "input_images"),
        os.path.join(project_root, "dataset", "get_test_imgs")
    ]
    
    for path in possible_paths:
        if os.path.isdir(path) and has_image_files(path):
            print(f"Found input directory with images at: {path}")
            return path
    
    # If no directory with images is found, return the original input directory
    print(f"No directory with images found. Using default: {input_dir}")
    return input_dir

def has_image_files(directory, exts=None):
    """Check if a directory contains image files with given extensions"""
    if exts is None:
        exts = ['jpg', 'jpeg', 'png']
    
    for ext in exts:
        pattern = os.path.join(directory, f'*.{ext}')
        if glob.glob(pattern):
            return True
        pattern = os.path.join(directory, f'*.{ext.upper()}')
        if glob.glob(pattern):
            return True
    
    return False

def main():
    """Main function to process all images in a directory"""
    # Parse arguments
    args = parse_arguments()
    
    # Get project root directory from current script location
    core_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(core_dir)
    
    # Find the input directory, using project_root for relative paths
    input_dir = find_input_directory(args.input_dir, project_root)
    
    # Set up output directory in project root
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
    
    # Find all image files in the input directory
    extensions = args.extensions.split(',')
    image_files = set()  # Use a set to avoid duplicates
    
    print(f"Searching for images with extensions: {extensions} in {input_dir}")
    for ext in extensions:
        pattern = os.path.join(input_dir, f'*.{ext}')
        found = glob.glob(pattern)
        for f in found:
            image_files.add(f)
        print(f"Found {len(found)} files with extension .{ext}")
        
        pattern = os.path.join(input_dir, f'*.{ext.upper()}')
        found = glob.glob(pattern)
        for f in found:
            image_files.add(f)
        print(f"Found {len(found)} files with extension .{ext.upper()}")
    
    # Convert set back to sorted list
    image_files = sorted(list(image_files))
    
    if not image_files:
        print(f"No images found in {input_dir} with extensions {args.extensions}")
        return
    
    # Limit the number of images if max_images is set
    if args.max_images is not None and args.max_images > 0:
        image_files = image_files[:args.max_images]
        print(f"Limiting to {args.max_images} images")
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    success_count = 0
    failure_count = 0
    start_time = time.time()
    
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing {os.path.basename(image_path)}")
        
        try:
            success = process_image(image_path, args)
            if success:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            failure_count += 1    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Batch Processing Complete!")
    print(f"Processed {len(image_files)} images in {total_time:.1f} seconds")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Input directory: {input_dir}")
      # Create a summary file with all the processed images
    summary_path = os.path.join(project_root, "batch_processing_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Batch Processing Summary\n")
        f.write(f"======================\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Directory: {input_dir}\n")
        f.write(f"Images Processed: {len(image_files)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {failure_count}\n")
        f.write(f"Total Time: {total_time:.1f} seconds\n\n")
        
        f.write("Processed Images:\n")
        for i, image_path in enumerate(image_files):
            basename = os.path.basename(image_path)
            safe_name = os.path.splitext(basename)[0].replace('.', '_').replace(' ', '_')
            output_dir = os.path.join(project_root, "outputs", safe_name)
            
            status = "Success" if os.path.isdir(output_dir) else "Failed"
            f.write(f"{i+1}. {basename}: {status}\n")
    
    # Print information about the output structure
    outputs_dir = os.path.join(project_root, "outputs")
    if os.path.isdir(outputs_dir):
        print("\nOutput Structure:")
        processed_files = set()
        for item in os.listdir(outputs_dir):
            item_path = os.path.join(outputs_dir, item)
            if os.path.isdir(item_path):
                print(f"  - {item}/")
                files = os.listdir(item_path)
                print(f"    {len(files)} files")
                processed_files.add(item)
        print(f"\nTotal unique images processed: {len(processed_files)}")
    else:
        print("\nNo 'outputs' directory found. Check for errors.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
