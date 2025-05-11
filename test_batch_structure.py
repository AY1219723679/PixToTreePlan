#!/usr/bin/env python3
"""
Test script for batch processing with the new folder structure

This script simply calls batch_process.py with the proper parameters.
"""

import os
import sys
import subprocess

def main():
    """Main function"""
    
    # Build command
    cmd = [
        sys.executable,
        "batch_process.py",
        "--input_dir=dataset/input_images",
        "--max_images=1",
        "--visualize"
    ]
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # Check outputs directory
    if os.path.isdir("outputs"):
        print("\nContents of outputs directory:")
        for item in os.listdir("outputs"):
            item_path = os.path.join("outputs", item)
            if os.path.isdir(item_path):
                print(f"  - {item}/")
                for file in os.listdir(item_path):
                    file_size = os.path.getsize(os.path.join(item_path, file)) / 1024  # KB
                    print(f"    - {file} ({file_size:.1f} KB)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
