#!/usr/bin/env python3
"""
Run script for stump center visualization

This script runs the visualization of 3D stump center points 
on the ground point cloud with appropriate parameters.
"""

import os
import sys
import subprocess

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

def main():
    # Define the paths for visualization
    ply_path = os.path.join(project_root, "outputs", "point_cloud.ply")
    
    # Check if the default path exists
    if not os.path.exists(ply_path):
        # Try to find a suitable PLY file in the outputs directory
        outputs_dir = os.path.join(project_root, "outputs")
        found = False
        
        for root, dirs, files in os.walk(outputs_dir):
            for file in files:
                if file.endswith(".ply"):
                    ply_path = os.path.join(root, file)
                    found = True
                    print(f"Found PLY file: {ply_path}")
                    break
            if found:
                break
                
        if not found:
            print("Error: Could not find a PLY file in the outputs directory.")
            print("Please specify the path to a PLY file using the --ply argument.")
            sys.exit(1)
    
    # Set up the command to run the visualization script
    vis_script = os.path.join(project_root, "visualize_stump_centers.py")
    cmd = [
        sys.executable,  # Use the current Python interpreter
        vis_script,
        f"--ply={ply_path}",
        "--point_size=1.5",
        "--center_size=100",
        "--title=3D Stump Centers on Ground Point Cloud"
    ]
    
    print("Running visualization script with the following command:")
    print(" ".join(cmd))
    
    # Execute the visualization script
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error running visualization script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
