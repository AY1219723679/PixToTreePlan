#!/usr/bin/env python3
"""
Test Script for the New Folder Structure

This script processes a single image to test the new folder structure
where all outputs for an image are stored in a subfolder named after the image.
"""

import os
import sys
import subprocess
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test the new folder structure for PixToTreePlan')
    
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the input image')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations for each step')
    
    return parser.parse_args()

def find_test_image():
    """Find a test image to use"""
    # Try different possible locations
    possible_paths = [
        os.path.join("dataset", "input_images"),
        "input_images"
    ]
    
    for path in possible_paths:
        if os.path.isdir(path):
            # Look for jpg files
            for file in os.listdir(path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(path, file)
    
    return None

def main():
    """Main function"""
    args = parse_arguments()
    
    # Get the absolute path to the script directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # If no image provided, find a test image
    if args.image_path is None:
        args.image_path = find_test_image()
        if args.image_path is None:
            print("Error: No test image found and no image path provided")
            return 1
        print(f"Using test image: {args.image_path}")
    
    # Build command to process this image
    cmd = [
        sys.executable,
        os.path.join(project_root, 'main.py'),
        f'--image_path={args.image_path}',
    ]
    
    if args.visualize:
        cmd.append('--visualize')
    
    # Call the main script
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # Check if output directory exists
    image_basename = os.path.splitext(os.path.basename(args.image_path))[0]
    output_dir = os.path.join("outputs", image_basename.replace('.', '_').replace(' ', '_'))
    
    if os.path.isdir(output_dir):
        print(f"\nOutput files in {output_dir}:")
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"  - {file} ({file_size:.1f} KB)")
        print("\nTest successful! Output files were created in the correct directory.")
    else:
        print(f"\nError: Output directory {output_dir} was not created.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
