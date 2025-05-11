#!/usr/bin/env python3
"""
Clean up temporary and legacy files script

This script removes any temporary image files from the root directory
and ensures that all outputs are properly organized in the outputs/ structure.
"""

import os
import glob
import shutil

# Files that might exist in the root directory and should be removed
ROOT_FILES_TO_CLEAN = [
    "cutout_ground.png",
    "ground_mask.png",
    "temp_for_depth.jpg",
    "depth_map.png",
    "depth_masked.png",
    "depth_visualization.png"
]

# Legacy output directories
LEGACY_DIRS = [
    "output_depth",
    "output_groundmasks",
    "output_pointcloud"
]

def clean_root_files():
    """Clean up temporary files in the root directory"""
    
    print("Cleaning up temporary files in root directory...")
    
    # Check and remove each file
    for file_pattern in ROOT_FILES_TO_CLEAN:
        # Use glob to find any variations
        matching_files = glob.glob(file_pattern)
        for file in matching_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"  ✓ Removed {file}")
                except Exception as e:
                    print(f"  ✗ Failed to remove {file}: {e}")

def handle_legacy_dirs(keep=True):
    """Handle legacy output directories"""
    
    if keep:
        # Keep the directories but warn user
        print("\nKeeping legacy output directories for backward compatibility:")
        for dir_name in LEGACY_DIRS:
            if os.path.exists(dir_name):
                print(f"  - {dir_name}/ (legacy files will remain)")
    else:
        # Delete the legacy directories
        print("\nRemoving legacy output directories:")
        for dir_name in LEGACY_DIRS:
            if os.path.exists(dir_name):
                try:
                    shutil.rmtree(dir_name)
                    print(f"  ✓ Removed {dir_name}/")
                except Exception as e:
                    print(f"  ✗ Failed to remove {dir_name}/: {e}")

def main():
    print("========================================")
    print("PixToTreePlan File Cleanup")
    print("========================================")
    
    # Clean root files
    clean_root_files()
    
    # Prompt for legacy directory handling
    print("\nLegacy output directories contain files from previous runs.")
    print("These are no longer used by the current version of the code.")
    choice = input("Do you want to remove legacy output directories? (y/n): ").strip().lower()
    
    if choice == 'y' or choice == 'yes':
        handle_legacy_dirs(keep=False)
    else:
        handle_legacy_dirs(keep=True)
    
    print("\nEnsuring outputs directory exists...")
    os.makedirs("outputs", exist_ok=True)
    
    print("\nCleanup complete!")
    print("All new outputs will be organized in the outputs/ directory")
    print("========================================")

if __name__ == "__main__":
    main()
