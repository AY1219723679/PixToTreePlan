#!/usr/bin/env python3
"""
This script will download and set up the DeepLabV3Plus-Pytorch repository
to fix the segmentation issues in PixToTreePlan.
"""

import os
import sys
import shutil
import subprocess
import zipfile
import tempfile
from urllib.request import urlretrieve
import platform

def main():
    # Get the current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {curr_dir}")
    
    # Define paths
    deeplab_path = os.path.join(curr_dir, "DeepLabV3Plus-Pytorch")
    get_ground_mask_path = os.path.join(curr_dir, "main", "get_ground_mask")
    deeplab_target_path = os.path.join(get_ground_mask_path, "DeepLabV3Plus-Pytorch")
    
    # Clean up existing directories if needed
    if os.path.exists(deeplab_path):
        print(f"Removing existing DeepLabV3Plus-Pytorch directory: {deeplab_path}")
        shutil.rmtree(deeplab_path)
    
    if os.path.exists(deeplab_target_path):
        print(f"Removing existing target directory: {deeplab_target_path}")
        shutil.rmtree(deeplab_target_path)
    
    # URL for the DeepLabV3Plus-Pytorch repository zip
    repo_url = "https://github.com/VainF/DeepLabV3Plus-Pytorch/archive/refs/heads/master.zip"
    
    print(f"Downloading DeepLabV3Plus-Pytorch from {repo_url}")
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
        shutil.move(extracted_dir, deeplab_path)
        print(f"Extracted to {deeplab_path}")
        
        # Copy to the target path in main/get_ground_mask/
        os.makedirs(os.path.dirname(deeplab_target_path), exist_ok=True)
        shutil.copytree(deeplab_path, deeplab_target_path)
        print(f"Copied to {deeplab_target_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
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
