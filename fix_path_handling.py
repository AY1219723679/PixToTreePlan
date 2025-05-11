#!/usr/bin/env python3
"""
Path Fixing Script

This script modifies the code in main.py to handle file paths properly,
without requiring a wrapper script.
"""

import os
import re
import sys

def fix_main_py():
    """Fix step2_create_cutout in main.py to handle absolute image paths"""
    main_py_path = "main.py"
    if not os.path.exists(main_py_path):
        print(f"Error: {main_py_path} not found")
        return False
    
    with open(main_py_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find and modify the create_cutout function
    pattern = r'def step2_create_cutout\(image_path, mask_path, image_basename\):(.*?)return cutout_path'
    replacement = '''def step2_create_cutout(image_path, mask_path, image_basename):
    """
    Step 2: Create a cutout image using the ground mask
    """
    print("\\nSTEP 2: Creating cutout image...")
    
    # Create an absolute path copy of the image if needed
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    
    # Create cutout
    cutout_path = "cutout_ground.png"
    create_cutout_with_mask(image_path, mask_path, cutout_path)
    
    # Also save a copy in output_groundmasks with the original image name
    output_cutout_path = os.path.join("output_groundmasks", f"{image_basename}_cutout.png")
    shutil.copy(cutout_path, output_cutout_path)
    
    print(f"  Cutout saved to {cutout_path}")
    print(f"  Copy saved to {output_cutout_path}")
    
    return cutout_path'''
    
    modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if modified_content == content:
        print("No changes made to main.py - pattern not found")
        return False
    
    with open(main_py_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)
    
    print("Successfully updated main.py to handle absolute image paths")
    return True

def main():
    """Main function"""
    # Get the current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the project root directory
    if not os.path.exists("main.py") or not os.path.isdir("main"):
        print("Error: This script must be run from the project root directory")
        return 1
    
    # Fix the main.py file
    if not fix_main_py():
        print("Failed to fix path handling in main.py")
        return 1
    
    print("Path handling fixes applied successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
