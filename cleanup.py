#!/usr/bin/env python3
"""
Cleanup Script for PixToTreePlan

This script removes all the existing output files in preparation for
the new folder structure implementation.
"""

import os
import shutil

def clean_output_directories():
    """Clean all output directories"""
    print("Cleaning up output directories...")
    
    # Directories to clean
    directories = [
        "output_groundmasks",
        "output_depth",
        "output_pointcloud",
    ]
    
    # Files to remove
    individual_files = [
        "ground_mask.png",
        "cutout_ground.png",
    ]
    
    # Clean directories
    for directory in directories:
        if os.path.exists(directory):
            print(f"Cleaning {directory}...")
            shutil.rmtree(directory)
            # Recreate empty directory
            os.makedirs(directory, exist_ok=True)
    
    # Remove individual files
    for file_path in individual_files:
        if os.path.exists(file_path):
            print(f"Removing {file_path}...")
            os.remove(file_path)
    
    print("Cleanup complete!")

if __name__ == "__main__":
    clean_output_directories()
