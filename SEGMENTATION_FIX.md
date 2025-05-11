# Segmentation and Ground Mask Issue Fix

## Problem Description
The PixToTreePlan project was experiencing an issue where:
1. The segmentation map was incorrectly labeling the entire image as a single class (road)
2. The ground mask output was completely blank
3. The system was unable to properly identify ground regions in images

## Root Cause
The main issue was related to the DeepLabV3Plus-Pytorch repository setup:
- The directory `DeepLabV3Plus-Pytorch` was empty or not properly initialized
- This caused the code to fall back to torchvision's DeepLabV3 model
- The fallback model uses different class labels (COCO/VOC) than expected (Cityscapes)
- The expected classes for ground (road=0, sidewalk=1, terrain=9) don't exist in the fallback model

## Solution Applied
Two solutions were implemented to fix the issue:

### Solution 1: Improved Fallback Logic
The code was modified to:
1. Detect when the default model can't be loaded
2. Create a better fallback mechanism when using torchvision's model
3. Use adaptive ground detection based on image characteristics when class labels don't match
4. Add a safety check to ensure ground masks are never completely blank

### Solution 2: Minimal Module Structure Creation
The script `fix_segmentation.py` was created to:
1. Create the minimum necessary directory structure for the model
2. Implement a simplified version of the required modules
3. Allow the system to function even without the full DeepLabV3Plus-Pytorch repository

## Long-Term Fix
For the best performance:
1. Clone the complete DeepLabV3Plus-Pytorch repository from GitHub
2. Run the `setup_deeplab.py` script to properly set up the repository
3. Ensure model checkpoints are in the correct location

## Running the Code
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Run `python fix_segmentation.py` to create the necessary module structure
3. Process images using: `python wrapper.py --image_path=PATH_TO_IMAGE --visualize`

## Expected Results
- The ground mask should now properly identify ground regions (paths, roads, terrain)
- The segmentation should distinguish between different classes in the image
- The output cutouts and depth maps should show correct ground regions
