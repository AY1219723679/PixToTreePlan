#!/usr/bin/env python3
"""
Fix batch_process.py to work with the new output folder structure

This script:
1. Updates batch_process.py to handle the new output folder structure
2. Makes sure it correctly reports the location of output files
"""

import os
import re
import sys
import shutil

def fix_batch_process():
    """Fix the batch_process.py file"""
    batch_process_path = "batch_process.py"
    print(f"Fixing {batch_process_path}...")
    
    # Read the entire file contents
    try:
        with open(batch_process_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"Error reading {batch_process_path}: {e}")
        return False
    
    # Create a backup of the original file
    backup_path = "batch_process.py.bak"
    try:
        shutil.copy2(batch_process_path, backup_path)
        print(f"Created backup at {backup_path}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Update the content with the new folder structure support
    updated_content = original_content
    
    # Find and update the process_image function to check for outputs in the new location
    process_image_pattern = r"(# Check if output directory exists\s+)([^#]+?)(\s+# Return success or failure)"
    
    new_check_code = """    # Check if output directory exists
    output_dir = os.path.join("outputs", image_basename.replace('.', '_').replace(' ', '_'))
    if os.path.isdir(output_dir):
        print(f"Output directory exists: {output_dir}")
        print(f"Files created: {len(os.listdir(output_dir))}")
        
        # Check for critical files
        critical_files = ["ground_mask.png", "depth_map.png", "cutout.png"]
        missing_files = []
        for file in critical_files:
            if not os.path.exists(os.path.join(output_dir, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"Warning: Missing critical files: {', '.join(missing_files)}")
            return False
        
        print(f"Output saved to: {output_dir}")
        return True
    else:
        print(f"Error: Output directory not created: {output_dir}")
        return False"""
    
    updated_content = re.sub(process_image_pattern, 
                             lambda m: m.group(1) + new_check_code + m.group(3),
                             original_content, 
                             flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(batch_process_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Updated {batch_process_path} to work with the new output folder structure")
    return True

def main():
    """Main function"""
    print("Fixing batch_process.py to work with the new output folder structure...")
    
    # Fix batch_process.py
    if not fix_batch_process():
        print("Failed to fix batch_process.py")
        return 1
    
    print("\nBatch processing script updated successfully!")
    print("The script will now properly handle the new output folder structure.")
    
    # Show how to test the changes
    print("\nTest the changes with:")
    print("  python batch_process.py --max_images=2")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
