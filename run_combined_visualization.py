#!/usr/bin/env python3
"""
Run the combined point cloud and center point visualization
with an example from the project.
"""

import os
import sys
import subprocess

def main():
    # Set up paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Look for a sample PLY file in outputs
    sample_ply = None
    outputs_dir = os.path.join(project_root, "outputs")
    
    # Default test case
    default_folder = os.path.join(outputs_dir, "urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab")
    default_ply = os.path.join(default_folder, "point_cloud.ply")
    
    if os.path.exists(default_ply):
        sample_ply = default_ply
        print(f"Using default PLY file: {sample_ply}")
    else:
        # Search for any PLY file in outputs
        for root, dirs, files in os.walk(outputs_dir):
            for file in files:
                if file.endswith(".ply"):
                    sample_ply = os.path.join(root, file)
                    print(f"Found PLY file: {sample_ply}")
                    break
            if sample_ply:
                break
    
    if not sample_ply:
        print("Error: Could not find a PLY file to visualize.")
        sys.exit(1)
    
    # Default YOLO label for tree example
    yolo_label = os.path.join(project_root, "YOLO", "train", "labels", "urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.txt")
      # Run the visualization script
    cmd = [
        sys.executable,
        os.path.join(project_root, "visualize_combined_pointclouds.py"),
        f"--ply={sample_ply}",
        "--point_size=2.0",  # Ensure point cloud points are visible
        "--center_size=10.0"  # Make center points stand out
    ]
    
    # Add the image path
    image_path = os.path.join(os.path.dirname(sample_ply), "original.png")
    if os.path.exists(image_path):
        cmd.append(f"--image={image_path}")
    
    # Add the depth map path
    depth_path = os.path.join(os.path.dirname(sample_ply), "depth_map.png")
    if os.path.exists(depth_path):
        cmd.append(f"--depth={depth_path}")
    
    # Add the YOLO label path
    if os.path.exists(yolo_label):
        cmd.append(f"--label={yolo_label}")
    
    # Add output path
    output_path = os.path.join(project_root, "combined_visualization.html")
    cmd.append(f"--output={output_path}")
    
    # Run the visualization script
    print("Running visualization...")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
