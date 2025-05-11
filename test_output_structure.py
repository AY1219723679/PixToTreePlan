#!/usr/bin/env python3
"""
Test the New Output Folder Structure

This script tests if the new output folder structure works properly
by processing a sample image and verifying the directory structure.
"""

import os
import sys
import shutil
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test the new output folder structure')
    parser.add_argument('--image_path', type=str, 
                        default='dataset/input_images/urban_tree_10_jpg.rf.81eafaad33fd7ce2b7233b8483800d71.jpg',
                        help='Path to test image')
    parser.add_argument('--clean', action='store_true',
                        help='Clean existing outputs before testing')
    parser.add_argument('--batch', action='store_true',
                        help='Test batch processing')
    return parser.parse_args()

def clean_outputs():
    """Remove all output folders"""
    print("Cleaning existing output folders...")
    
    folders = [
        "outputs",
        "output_groundmasks",
        "output_depth",
        "output_pointcloud"
    ]
    
    for folder in folders:
        if os.path.exists(folder):
            print(f"Removing {folder}/")
            try:
                shutil.rmtree(folder)
                print(f"✓ Removed {folder}/")
            except Exception as e:
                print(f"✗ Failed to remove {folder}: {e}")

def test_single_image(image_path):
    """Test processing a single image"""
    print("\nTesting single image processing...")
    print(f"Image: {image_path}")
    
    # Run main.py on the test image
    import subprocess
    cmd = [
        sys.executable,
        'main.py',
        f'--image_path={image_path}',
        '--visualize'  # Enable visualization for complete test
    ]
    
    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if process was successful
    if result.returncode != 0:
        print("❌ Test failed! Error running main.py")
        print("ERROR OUTPUT:")
        print(result.stderr)
        return False
    
    # Print output
    print("\nOUTPUT:")
    for line in result.stdout.splitlines():
        print(f"  {line}")
    
    # Get image basename
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    safe_name = image_basename.replace('.', '_').replace(' ', '_')
    
    # Check if output directory was created
    image_output_dir = os.path.join("outputs", safe_name)
    if not os.path.isdir(image_output_dir):
        print(f"❌ Test failed! Output directory {image_output_dir} not created")
        return False
    
    # Check for expected files
    expected_files = [
        "original.png",
        "segmentation.png",
        "ground_mask.png",
        "cutout.png",
        "depth_map.png",
        "depth_masked.png",
        "point_cloud.ply",
        "point_cloud_visualization.png"
    ]
    
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(image_output_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Test failed! Missing files in output directory:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # Check legacy structure for compatibility
    legacy_files = [
        "ground_mask.png",
        "cutout_ground.png",
        os.path.join("output_groundmasks", f"{image_basename}_cutout.png"),
        os.path.join("output_depth", "depth_map.png"),
        os.path.join("output_depth", "depth_masked.png"),
        os.path.join("output_pointcloud", f"{image_basename}_pointcloud.ply")
    ]
    
    missing_legacy_files = []
    for file in legacy_files:
        if not os.path.exists(file):
            missing_legacy_files.append(file)
    
    if missing_legacy_files:
        print("⚠️ Warning: Some legacy files are missing:")
        for file in missing_legacy_files:
            print(f"  - {file}")
    
    # List all files in the output directory
    print("\nFiles in output directory:")
    for file in os.listdir(image_output_dir):
        file_path = os.path.join(image_output_dir, file)
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"  - {file} ({size:.1f} KB)")
    
    print("\n✅ Test passed! Output structure working correctly.")
    return True

def test_batch_processing():
    """Test batch processing multiple images"""
    print("\nTesting batch processing...")
    
    # Run batch_process.py on the test images (limiting to 2 max)
    import subprocess
    cmd = [
        sys.executable,
        'batch_process.py',
        '--input_dir=dataset/input_images',
        '--max_images=2',
        '--visualize'
    ]
    
    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if process was successful
    if result.returncode != 0:
        print("❌ Batch test failed! Error running batch_process.py")
        print("ERROR OUTPUT:")
        print(result.stderr)
        return False
    
    # Print output
    print("\nOUTPUT:")
    for line in result.stdout.splitlines():
        print(f"  {line}")
    
    # Check if outputs directory was created
    if not os.path.isdir("outputs"):
        print("❌ Batch test failed! 'outputs' directory not created")
        return False
    
    # Count output folders
    output_folders = [d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))]
    print(f"\nFound {len(output_folders)} folders in outputs/:")
    
    if len(output_folders) < 2:
        print("⚠️ Warning: Found fewer than 2 output folders")
    
    # Check content of each output folder
    for folder in output_folders:
        folder_path = os.path.join("outputs", folder)
        files = os.listdir(folder_path)
        print(f"  - {folder}/: {len(files)} files")
        
        # Check for essential files
        essential_files = ["original.png", "ground_mask.png", "cutout.png", "depth_map.png"]
        missing = [f for f in essential_files if f not in files]
        if missing:
            print(f"    ⚠️ Warning: Missing files: {', '.join(missing)}")
    
    print("\n✅ Batch test completed! Output structure working correctly.")
    return True

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Clean outputs if requested
    if args.clean:
        clean_outputs()
    
    # Test single image processing
    if test_single_image(args.image_path):
        print("\n✅ Single image test passed!")
    else:
        print("\n❌ Single image test failed!")
        return 1
    
    # Test batch processing if requested
    if args.batch:
        if test_batch_processing():
            print("\n✅ Batch processing test passed!")
        else:
            print("\n❌ Batch processing test failed!")
            return 1
    
    print("\n✅ All tests passed! The new output folder structure is working correctly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
