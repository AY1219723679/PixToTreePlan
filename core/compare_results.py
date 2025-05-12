#!/usr/bin/env python3
"""
Compare Results Script

This script creates a visualization grid comparing the results from multiple processed images,
showing the original image, ground mask, depth map, and point cloud visualization side-by-side.
"""

import os
import sys
import argparse
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate comparison visualizations for processed images')
    
    parser.add_argument('--image_dir', type=str, default='input_images',
                        help='Directory containing original input images')
    parser.add_argument('--extension', type=str, default='jpg',
                        help='File extension of images to compare')
    parser.add_argument('--max_images', type=int, default=10,
                        help='Maximum number of images to include in comparison')
    parser.add_argument('--output_path', type=str, default='comparison_results.png',
                        help='Path to save the comparison visualization')
    parser.add_argument('--include_pointcloud', action='store_true',
                        help='Include point cloud visualizations in the comparison')
    
    return parser.parse_args()

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

def find_processed_images(image_dir, extension, max_images):
    """Find images that have been fully processed"""
    # Get all original images
    original_images = sorted(glob.glob(os.path.join(image_dir, f'*.{extension}')))[:max_images]
    
    # Check which ones have all the required outputs
    processed_images = []
    
    for img_path in original_images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Check for ground mask
        mask_path = os.path.join('output_groundmasks', f'{basename}_groundmask.png')
        
        # Check for depth map
        depth_path = os.path.join('output_depth', 'depth_masked.png')
        
        # Check for point cloud visualization (optional)
        pointcloud_vis_path = os.path.join('output_pointcloud', f'{basename}_visualization.png')
        
        # If the required files exist, add to the list
        if os.path.exists(mask_path) and os.path.exists(depth_path):
            result = {
                'original': img_path,
                'mask': mask_path,
                'depth': depth_path,
                'pointcloud_vis': pointcloud_vis_path if os.path.exists(pointcloud_vis_path) else None
            }
            processed_images.append(result)
    
    return processed_images

def create_comparison_grid(processed_images, output_path, include_pointcloud):
    """Create a grid of images comparing the results"""
    n_images = len(processed_images)
    if n_images == 0:
        print("No processed images found!")
        return
    
    # Determine grid size
    n_cols = 3 if not include_pointcloud else 4
    n_rows = n_images
    
    # Create figure
    fig_width = n_cols * 4
    fig_height = n_rows * 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # If only one row, wrap axes in a list
    if n_rows == 1:
        axes = [axes]
    
    # Add column headers
    column_titles = ['Original', 'Ground Mask', 'Depth Map']
    if include_pointcloud:
        column_titles.append('Point Cloud')
    
    for j, title in enumerate(column_titles):
        fig.text(
            (j + 0.5) / n_cols, 0.98, 
            title, 
            ha='center', va='center',
            fontsize=16, fontweight='bold'
        )
    
    # Add images to the grid
    for i, img_data in enumerate(processed_images):
        # Extract image paths
        original_path = img_data['original']
        mask_path = img_data['mask']
        depth_path = img_data['depth']
        pointcloud_vis_path = img_data['pointcloud_vis']
        
        # Add image name as row label
        img_name = os.path.basename(original_path)
        fig.text(
            0.01, 1.0 - (i + 0.5) / n_rows, 
            img_name, 
            ha='left', va='center',
            fontsize=10, fontweight='bold'
        )
        
        # Original image
        original_img = np.array(Image.open(original_path).convert('RGB'))
        axes[i][0].imshow(original_img)
        axes[i][0].axis('off')
        
        # Ground mask
        mask_img = np.array(Image.open(mask_path))
        axes[i][1].imshow(mask_img, cmap='gray')
        axes[i][1].axis('off')
        
        # Depth map
        depth_img = np.array(Image.open(depth_path))
        axes[i][2].imshow(depth_img, cmap='inferno')
        axes[i][2].axis('off')
        
        # Point cloud visualization (if available)
        if include_pointcloud and pointcloud_vis_path:
            try:
                pc_img = np.array(Image.open(pointcloud_vis_path))
                axes[i][3].imshow(pc_img)
                axes[i][3].axis('off')
            except Exception as e:
                print(f"Error loading point cloud visualization: {e}")
                axes[i][3].text(0.5, 0.5, 'No visualization', 
                                ha='center', va='center')
                axes[i][3].axis('off')
        elif include_pointcloud:
            axes[i][3].text(0.5, 0.5, 'Not available', 
                            ha='center', va='center')
            axes[i][3].axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison grid saved to: {output_path}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Find the input directory
    input_dir = find_input_directory(args.image_dir)
    
    print(f"Searching for processed images in {input_dir} with extension .{args.extension}")
    
    # Find processed images
    processed_images = find_processed_images(
        input_dir, 
        args.extension, 
        args.max_images
    )
    
    print(f"Found {len(processed_images)} fully processed images")
    
    # Create comparison grid
    create_comparison_grid(processed_images, args.output_path, args.include_pointcloud)

if __name__ == "__main__":
    main()
