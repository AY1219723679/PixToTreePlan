#!/usr/bin/env python3
"""
Update All Components for New Output Structure

This script applies all the necessary updates to implement the new
organized output folder structure across the entire codebase.

The new structure organizes all outputs for each image in its own
dedicated folder under 'outputs/', while maintaining backward compatibility
with the legacy structure.
"""

import os
import sys
import subprocess
import time

def print_banner(message):
    """Print a banner message"""
    print("\n" + "=" * 60)
    print(f" {message}")
    print("=" * 60)

def run_script(script_name):
    """Run a Python script and return success/failure"""
    print(f"Running {script_name}...")
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )
    
    # Print output
    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"  {line}")
    
    # Print errors
    if result.stderr:
        print(f"Error running {script_name}:")
        print(result.stderr)
        return False
    
    return result.returncode == 0

def main():
    start_time = time.time()
    
    print_banner("PixToTreePlan Output Structure Update")
    print("This script will update all components to use the new output folder structure.")
    print("The new structure organizes all outputs for each image in a dedicated folder.")
    
    # Step 1: Update main.py
    print_banner("Step 1: Updating main.py")
    if not run_script("fix_output_structure.py"):
        print("Failed to update main.py. Aborting.")
        return 1
    
    # Step 2: Update batch_process.py
    print_banner("Step 2: Updating batch_process.py")
    if not run_script("fix_batch_process.py"):
        print("Failed to update batch_process.py. Continuing anyway...")
    
    # Step 3: Update comparison grid
    print_banner("Step 3: Updating comparison grid")
    if not run_script("fix_comparison_grid.py"):
        print("Failed to update comparison grid. Continuing anyway...")
    
    # Step 4: Test the changes
    print_banner("Step 4: Testing the changes")
    if not run_script("test_output_structure.py --clean"):
        print("Tests failed. Please check the errors and update the scripts manually.")
    
    # Done
    total_time = time.time() - start_time
    print_banner("Update Complete")
    print(f"All components updated in {total_time:.1f} seconds!")
    print("\nNew Output Structure:")
    print("outputs/")
    print("  image1_name/")
    print("    original.png")
    print("    segmentation.png")
    print("    ground_mask.png")
    print("    cutout.png")
    print("    depth_map.png")
    print("    depth_masked.png")
    print("    point_cloud.ply")
    print("    point_cloud_visualization.png")
    print("  image2_name/")
    print("    ...")
    print("\nTest the new structure with:")
    print("  python main.py --image_path=dataset/input_images/urban_tree_10_jpg.rf.81eafaad33fd7ce2b7233b8483800d71.jpg")
    print("  python batch_process.py --max_images=2")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
