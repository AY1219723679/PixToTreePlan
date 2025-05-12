#!/usr/bin/env python3
"""
Modified setup script to download and set up the DeepLabV3Plus-Pytorch repository
with better error handling for permission issues.
"""

import os
import sys
import shutil
import subprocess
import zipfile
import tempfile
import time
from urllib.request import urlretrieve
import platform

def safely_remove_dir(path):
    """Safely remove a directory with better error handling for permission issues"""
    if not os.path.exists(path):
        print(f"Directory does not exist: {path}")
        return True
    
    print(f"Attempting to remove directory: {path}")
    try:
        shutil.rmtree(path)
        return True
    except PermissionError as e:
        print(f"Permission error: {e}")
        print("Trying alternative approach...")
        try:
            # Try using system commands as fallback
            if platform.system() == "Windows":
                os.system(f'rd /s /q "{path}"')
            else:
                os.system(f'rm -rf "{path}"')
            time.sleep(2)  # Give some time for the OS to complete
            return not os.path.exists(path)
        except Exception as e2:
            print(f"Failed to remove directory: {e2}")
            return False

def main():
    # Get the current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {curr_dir}")
    
    # Define paths
    deeplab_path = os.path.join(curr_dir, "DeepLabV3Plus-Pytorch")
    get_ground_mask_path = os.path.join(curr_dir, "main", "get_ground_mask")
    deeplab_target_path = os.path.join(get_ground_mask_path, "DeepLabV3Plus-Pytorch")
    
    # Create directories if they don't exist
    os.makedirs(get_ground_mask_path, exist_ok=True)
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(curr_dir, "..", "checkpoints", "best_deeplabv3plus_mobilenet_cityscapes_os16.pth")
    if not os.path.isfile(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        print("Segmentation results may not be accurate without the proper checkpoint.")
    else:
        print(f"Found checkpoint file at {checkpoint_path}")
    
    # Clean up existing directories if needed
    if os.path.exists(deeplab_path):
        if not safely_remove_dir(deeplab_path):
            print(f"Warning: Could not fully remove {deeplab_path}, continuing anyway")
    
    if os.path.exists(deeplab_target_path):
        if not safely_remove_dir(deeplab_target_path):
            print(f"Warning: Could not fully remove {deeplab_target_path}")
            print("You may need to manually remove this directory and try again.")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                print("Setup aborted.")
                return
    
    # URL for the DeepLabV3Plus-Pytorch repository zip
    repo_url = "https://github.com/VainF/DeepLabV3Plus-Pytorch/archive/refs/heads/master.zip"
    
    print(f"Downloading DeepLabV3Plus-Pytorch from {repo_url}")
    temp_dir = None
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "deeplabv3plus.zip")
        
        # Download the zip file
        urlretrieve(repo_url, zip_path)
        print(f"Downloaded to {zip_path}")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the extracted directory
        extracted_dir = None
        for item in os.listdir(temp_dir):
            if item.startswith("DeepLabV3Plus-Pytorch"):
                extracted_dir = os.path.join(temp_dir, item)
                break
        
        if not extracted_dir:
            print("Error: Could not find extracted DeepLabV3Plus-Pytorch directory")
            return
        
        # Move the extracted directory to the desired location
        if os.path.exists(deeplab_path):
            print(f"Warning: Target path {deeplab_path} already exists")
            safely_remove_dir(deeplab_path)
            
        print(f"Moving extracted files to {deeplab_path}")
        shutil.move(extracted_dir, deeplab_path)
        print(f"Extracted to {deeplab_path}")
        
        # Copy to the target path in main/get_ground_mask/
        os.makedirs(os.path.dirname(deeplab_target_path), exist_ok=True)
        
        print(f"Copying files to {deeplab_target_path}")
        if os.path.exists(deeplab_target_path):
            safely_remove_dir(deeplab_target_path)
            
        shutil.copytree(deeplab_path, deeplab_target_path)
        print(f"Copied to {deeplab_target_path}")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        return
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Install required packages
    print("\nInstalling required packages for DeepLabV3Plus-Pytorch")
    
    # Use the appropriate pip command based on the OS
    python_executable = sys.executable
    pip_command = [python_executable, "-m", "pip", "install", 
                  "torch", "torchvision", "tqdm", "Pillow", "numpy", "matplotlib"]
    
    try:
        subprocess.check_call(pip_command)
    except Exception as e:
        print(f"Error installing packages: {e}")
    
    print("\nDeepLabV3Plus-Pytorch setup complete!")
    print("\nYou should now be able to run PixToTreePlan with proper segmentation.")
    print("To test, run the following command:")
    
    if platform.system() == "Windows":
        print("    python main.py")
    else:
        print("    python3 main.py")
    
    print("\nIf you continue to have issues, please check the error messages and make sure")
    print("the checkpoint files are in the correct location.")

if __name__ == "__main__":
    main()
