#!/usr/bin/env python3
"""
Wrapper script for main.py that handles the image paths properly

This script copies the input image to the input_images/ directory temporarily
before calling the main.py script, and then cleans up afterwards.
"""

import os
import sys
import shutil
import argparse
import subprocess

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process an image with proper path handling')
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--resize_factor', type=float, default=0.6,
                        help='Resize factor for segmentation')
    parser.add_argument('--min_region_size', type=int, default=400,
                        help='Minimum size of regions to preserve')
    parser.add_argument('--min_area_ratio', type=float, default=0.1,
                        help='Min component size as percentage of image area')
    parser.add_argument('--z_scale', type=float, default=0.5,
                        help='Scale factor for Z values in point cloud')
    parser.add_argument('--sample_rate', type=int, default=2,
                        help='Sample rate for point cloud generation')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations for each step')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Get absolute paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_root, "input_images")
    os.makedirs(input_dir, exist_ok=True)
    
    # Get the original image path and basename
    src_image_path = os.path.abspath(args.image_path)
    image_basename = os.path.basename(src_image_path)
    
    # Target path in input_images directory
    target_image_path = os.path.join(input_dir, image_basename)
    
    # Copy the image to input_images/
    temp_file_created = False
    try:
        if src_image_path != target_image_path:
            print(f"Copying {src_image_path} to {target_image_path}")
            shutil.copy2(src_image_path, target_image_path)
            temp_file_created = True
        
        # Build command to call main.py with the copied image
        cmd = [
            sys.executable,
            os.path.join(project_root, "main.py"),
            f"--image_path={target_image_path}",
            f"--resize_factor={args.resize_factor}",
            f"--min_region_size={args.min_region_size}",
            f"--min_area_ratio={args.min_area_ratio}",
            f"--z_scale={args.z_scale}",
            f"--sample_rate={args.sample_rate}"
        ]
        
        if args.visualize:
            cmd.append("--visualize")
        
        # Call the main script
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        # Return the same exit code as the main script
        return result.returncode
    
    finally:
        # Clean up the temporary copy
        if temp_file_created and os.path.exists(target_image_path):
            print(f"Cleaning up temporary file: {target_image_path}")
            try:
                os.remove(target_image_path)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file: {e}")

if __name__ == "__main__":
    sys.exit(main())
