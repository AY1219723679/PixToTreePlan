#!/usr/bin/env python3
"""
Wrapper script for PixToTreePlan single image processing
This script ensures proper import paths and provides a simple interface to main.py
"""

import os
import sys
import argparse

def main():
    # Set up project paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    core_dir = os.path.join(current_dir, "core")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process a single image through the PixToTreePlan pipeline')
    
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the input image')
    parser.add_argument('--resize_factor', type=float, default=0.6,
                       help='Resize factor for segmentation (lower = coarser, range: 0.3-1.0)')
    parser.add_argument('--min_region_size', type=int, default=400,
                       help='Minimum size of regions to preserve (in pixels)')
    parser.add_argument('--min_area_ratio', type=float, default=0.1,
                       help='Min component size as percentage of image area (0.1 = 10%)')
    parser.add_argument('--z_scale', type=float, default=0.5,
                       help='Scale factor for Z values in point cloud')
    parser.add_argument('--sample_rate', type=int, default=2,
                       help='Sample rate for point cloud generation (1=full density)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations for each step')
                       
    args = parser.parse_args()
    
    # Build command to call main.py
    cmd = [
        sys.executable,
        os.path.join(core_dir, "main.py"),
        f"--image_path={args.image_path}",
        f"--resize_factor={args.resize_factor}",
        f"--min_region_size={args.min_region_size}",
        f"--min_area_ratio={args.min_area_ratio}",
        f"--z_scale={args.z_scale}",
        f"--sample_rate={args.sample_rate}"
    ]
    
    if args.visualize:
        cmd.append("--visualize")
    
    # Execute main.py with the PYTHONPATH set
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir  # Add project root to PYTHONPATH
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
