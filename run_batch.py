#!/usr/bin/env python3
"""
Wrapper script for PixToTreePlan batch processing
This script ensures proper import paths and provides a simple interface to batch_process.py
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
        description='Process multiple images through the PixToTreePlan pipeline')
    
    parser.add_argument('--input_dir', type=str, default='dataset/input_images',
                       help='Directory containing input images')
    parser.add_argument('--extensions', type=str, default='jpg,jpeg,png',
                       help='File extensions to process (comma-separated)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations for each step')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process')
                       
    args = parser.parse_args()
    
    # Build command to call batch_process.py
    cmd = [
        sys.executable,
        os.path.join(core_dir, "batch_process.py"),
        f"--input_dir={args.input_dir}",
        f"--extensions={args.extensions}"
    ]
    
    if args.visualize:
        cmd.append("--visualize")
        
    if args.max_images is not None:
        cmd.append(f"--max_images={args.max_images}")
    
    # Execute batch_process.py with the PYTHONPATH set
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir  # Add project root to PYTHONPATH
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
