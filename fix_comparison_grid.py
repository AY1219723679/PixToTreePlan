#!/usr/bin/env python3
"""
Fix the comparison grid script to work with the new output folder structure

This script:
1. Updates compare_results.py to support the new output folder structure
2. Makes sure it can handle both old and new file locations
"""

import os
import re
import sys
import shutil

def fix_compare_results():
    """Fix the compare_results.py file"""
    file_path = "compare_results.py"
    print(f"Fixing {file_path}...")
    
    # Read the entire file contents
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    # Create a backup of the original file
    backup_path = "compare_results.py.bak"
    try:
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at {backup_path}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Replace the get_output_files function with one that supports the new structure
    new_function = """def get_output_files(image_basename):
    \"\"\"
    Find the output files for a given image, checking both new and old folder structures
    
    Args:
        image_basename: Base name of the image without extension
        
    Returns:
        dict: Dictionary with paths to output files
    \"\"\"
    results = {
        'original': None,
        'ground_mask': None,
        'segmentation': None,
        'cutout': None,
        'depth_map': None,
        'depth_masked': None,
        'point_cloud_viz': None
    }
    
    # First check the new folder structure (preferred)
    safe_name = image_basename.replace('.', '_').replace(' ', '_')
    image_output_dir = os.path.join("outputs", safe_name)
    
    if os.path.isdir(image_output_dir):
        print(f"Found new output directory for {image_basename}")
        
        # Define file paths in the new structure
        results['original'] = os.path.join(image_output_dir, "original.png")
        results['ground_mask'] = os.path.join(image_output_dir, "ground_mask.png")
        results['segmentation'] = os.path.join(image_output_dir, "segmentation.png")
        results['cutout'] = os.path.join(image_output_dir, "cutout.png")
        results['depth_map'] = os.path.join(image_output_dir, "depth_map.png")
        results['depth_masked'] = os.path.join(image_output_dir, "depth_masked.png")
        results['point_cloud_viz'] = os.path.join(image_output_dir, "point_cloud_visualization.png")
    else:
        # Check the legacy structure as fallback
        print(f"No new output directory found for {image_basename}, checking legacy paths...")
        
        # Try to find original from input_dir
        results['original'] = f"input_images/{image_basename}.jpg"
        if not os.path.exists(results['original']):
            results['original'] = f"input_images/{image_basename}.png"
        if not os.path.exists(results['original']):
            results['original'] = f"dataset/input_images/{image_basename}.jpg"
        if not os.path.exists(results['original']):
            results['original'] = f"dataset/input_images/{image_basename}.png"
        
        # Try to find other files in legacy locations
        results['ground_mask'] = "ground_mask.png"
        results['cutout'] = f"output_groundmasks/{image_basename}_cutout.png"
        if not os.path.exists(results['cutout']):
            results['cutout'] = "cutout_ground.png"
        
        results['depth_map'] = "output_depth/depth_map.png"
        results['depth_masked'] = "output_depth/depth_masked.png"
        results['point_cloud_viz'] = f"output_pointcloud/{image_basename}_visualization.png"
    
    # Filter out files that don't exist
    for key in list(results.keys()):
        if results[key] and not os.path.exists(results[key]):
            print(f"  Warning: File not found: {results[key]}")
            results[key] = None
            
    return results"""
    
    # Find and replace the existing function
    function_pattern = r"def get_output_files\(image_basename\):[^}]*?\n    return results"
    updated_content = re.sub(function_pattern, new_function, original_content, flags=re.DOTALL)
    
    # Replace the process_image_comparisons function to handle missing files better
    process_pattern = r"def process_image_comparisons.*?images\):"
    new_process_image = """def process_image_comparisons(image_files, args, include_pointcloud=False):
    \"\"\"
    Process images and generate comparison data
    
    Args:
        image_files: List of image paths to process
        args: Command-line arguments
        include_pointcloud: Whether to include point cloud visualizations
        
    Returns:
        list: List of processed image data dictionaries
    \"\"\"
    processed_images = []
    
    print(f"Processing {len(image_files)} images for comparison...")
    
    for image_path in image_files:"""
    
    updated_content = re.sub(process_pattern, new_process_image, updated_content, flags=re.DOTALL)
    
    # Fix the main function to check output directory
    main_check = """    # Check if outputs directory exists
    if not os.path.isdir('outputs') and not os.path.isdir('output_groundmasks'):
        print("Error: No processed outputs found.")
        print("Please process some images first using main.py or batch_process.py")
        return 1"""
    
    main_pattern = r"def main\(\):[^\n]*?\n    args = parse_arguments\(\)"
    replacement = lambda m: m.group(0) + "\n" + main_check
    updated_content = re.sub(main_pattern, replacement, updated_content)
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Updated {file_path} to work with the new output folder structure")
    return True

def main():
    """Main function"""
    print("Fixing compare_results.py to work with the new output folder structure...")
    
    # Fix compare_results.py
    if not fix_compare_results():
        print("Failed to fix compare_results.py")
        return 1
    
    print("\nComparison grid script updated successfully!")
    print("The script will now properly handle the new output folder structure.")
    
    # Show how to test the changes
    print("\nTest the changes with:")
    print("  python compare_results.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
