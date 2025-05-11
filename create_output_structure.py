#!/usr/bin/env python3
"""
Minimal script to create organized output folders for image processing
"""

import os
import shutil
import sys
import glob
import argparse

def create_output_structure(image_path):
    """
    Create the output folder structure for an image
    
    Args:
        image_path: Path to the image
        
    Returns:
        output_dir: Path to the output directory
    """
    # Make base outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Get image basename
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create a safe directory name
    safe_name = image_basename.replace('.', '_').replace(' ', '_')
    
    # Create output directory for this image
    output_dir = os.path.join("outputs", safe_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the original image to the output directory
    output_image_path = os.path.join(output_dir, "original.png")
    shutil.copy2(image_path, output_image_path)
    
    # Create empty placeholder files for demonstration
    placeholder_files = [
        "segmentation.png",
        "ground_mask.png",
        "cutout.png",
        "depth_map.png",
        "depth_masked.png"
    ]
    
    for filename in placeholder_files:
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(f"Placeholder for {filename}")
    
    print(f"Created output structure in: {output_dir}")
    print(f"Files created:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
    
    return output_dir

def process_image(image_path):
    """Process a single image"""
    print(f"Processing image: {image_path}")
    output_dir = create_output_structure(image_path)
    return output_dir

def process_directory(input_dir, max_images=None):
    """Process a directory of images"""
    # Find image files
    extensions = ['jpg', 'jpeg', 'png']
    image_files = []
    
    for ext in extensions:
        pattern = os.path.join(input_dir, f'*.{ext}')
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(input_dir, f'*.{ext.upper()}')
        image_files.extend(glob.glob(pattern))
    
    # Sort and limit
    image_files.sort()
    if max_images:
        image_files = image_files[:max_images]
        
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing {os.path.basename(image_path)}")
        process_image(image_path)
    
    print("\nBatch processing complete!")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test script for creating output folder structure')
    
    parser.add_argument('--input_dir', type=str, default='dataset/input_images',
                        help='Directory containing input images')
    parser.add_argument('--max_images', type=int, default=2,
                        help='Maximum number of images to process')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found")
        return 1
    
    process_directory(args.input_dir, args.max_images)
    return 0

if __name__ == "__main__":
    sys.exit(main())
