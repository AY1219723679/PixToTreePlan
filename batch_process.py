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
    
    # Get the absolute path to the script directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.abspath(image_path)
    
    # Build command to process this image
    cmd = [
        sys.executable,
        os.path.join(project_root, 'main.py'),
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output and error (if any)
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"Error processing {image_path}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0

def find_input_directory(input_dir):
    """
    Find the correct input directory by checking multiple possible locations
    """
    # Check if the provided directory exists
    if os.path.isdir(input_dir):
        return input_dir
    
    # Try other common paths
    possible_paths = [
        input_dir,
        os.path.join("dataset", input_dir),
        "input_images",
        os.path.join("dataset", "input_images"),
        os.path.join("dataset", "get_test_imgs")
    ]
    
    for path in possible_paths:
        if os.path.isdir(path):
            print(f"Found input directory at: {path}")
            return path
    
    # If no directory is found, return the original (it will fail gracefully later)
    return input_dir

def main():
    """Main function to process all images in a directory"""
    # Parse arguments
    args = parse_arguments()
    
    # Find the input directory
    input_dir = find_input_directory(args.input_dir)
    
    # Find all image files in the input directory
    extensions = args.extensions.split(',')
    image_files = []
    
    for ext in extensions:
        pattern = os.path.join(input_dir, f'*.{ext}')
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(input_dir, f'*.{ext.upper()}')
        image_files.extend(glob.glob(pattern))
      # Sort the files
    image_files.sort()
    
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
            failure_count += 1
      # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Batch Processing Complete!")
    print(f"Processed {len(image_files)} images in {total_time:.1f} seconds")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Input directory: {input_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
