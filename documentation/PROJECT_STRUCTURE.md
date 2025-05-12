# Project Structure and Code Organization

## Directory Structure

```
PixToTreePlan/
│
├── main.py                      # Main entry point script for single image processing
├── batch_process.py             # Script for processing multiple images in a directory
├── compare_results.py           # Script for generating side-by-side result comparisons
├── run_batch_process.ps1        # PowerShell script for batch processing on Windows
├── run_batch_process.bat        # Batch file alternative for Windows users
├── README.md                    # Project documentation
├── GETTING_STARTED.md           # Getting started guide
├── GETTING_STARTED_BATCH.md     # Guide for batch processing
├── PROJECT_STRUCTURE.md         # This file - project structure documentation
├── requirements.txt             # Project dependencies
│
├── checkpoints/                 # Pre-trained model weights
│   └── best_deeplabv3plus_mobilenet_cityscapes_os16.pth
│
├── DeepLabV3Plus-Pytorch/       # DeepLabV3+ implementation (used as a library)
│
├── main/                        # Core modules directory
│   ├── generate_depth/          # MiDaS depth map generation
│   │   ├── __init__.py
│   │   └── simple_depth_generator.py
│   │
│   ├── get_ground_mask/         # Semantic segmentation for ground masks
│   │   ├── __init__.py
│   │   ├── create_cutout_simple.py
│   │   └── single_image_processing.py
│   │
│   └── img_to_pointcloud/       # Point cloud generation from depth and cutout
│       ├── __init__.py
│       └── image_to_pointcloud.py
│
├── midas_model/                 # MiDaS model weights
│   └── model-small.pt
│
├── input_images/                # Default directory for input images
├── output_depth/                # Output directory for depth maps
├── output_groundmasks/          # Output directory for ground masks
└── output_pointcloud/           # Output directory for point clouds
```

## Module Descriptions

### 1. Main Scripts
- **`main.py`**: The primary orchestrator for single image processing:
  - Loads necessary modules
  - Parses command line arguments
  - Calls the individual processing steps in sequence
  - Manages output paths and directories
  
- **`batch_process.py`**: Processes multiple images from a directory:
  - Takes a directory path as input
  - Processes all images with specified extensions
  - Calls `main.py` for each image
  - Provides a summary of successes and failures
  
- **`compare_results.py`**: Generates visual comparisons:
  - Creates side-by-side visualizations of processing results
  - Shows original image, ground mask, depth map, and point cloud
  - Useful for comparing results across multiple images
  
- **`run_batch_process.ps1`**: PowerShell script for Windows users:
  - Automatically processes all images in the default directory
  - Generates visualizations for all steps
  - Creates a comparison grid of results

- **`run_batch_process.bat`**: Batch file alternative for Windows users:
  - Same functionality as the PowerShell script
  - Compatible with Windows Command Prompt

### 2. Ground Mask Generation (`main/get_ground_mask/`)
- `single_image_processing.py`: Uses DeepLabV3+ for semantic segmentation
- `create_cutout_simple.py`: Creates transparent cutout using the mask

### 3. Depth Map Generation (`main/generate_depth/`)
- `simple_depth_generator.py`: Uses MiDaS to estimate depth from images

### 4. Point Cloud Generation (`main/img_to_pointcloud/`)
- `image_to_pointcloud.py`: Combines cutout and depth map to create 3D point cloud

## Processing Pipeline

### Single Image Processing
1. **Image Input**: Takes an RGB image as input
2. **Segmentation**: Identifies ground-related classes (road, sidewalk, terrain)
3. **Mask Processing**: Creates and cleans binary ground mask
4. **Cutout Creation**: Applies mask to create transparent cutout
5. **Depth Estimation**: Generates depth map using MiDaS
6. **Point Cloud Creation**: Creates 3D point cloud using depth and RGB information
7. **Output**: Saves results in respective output directories

### Batch Processing
1. **Directory Scanning**: Finds all images with specified extensions
2. **Per-Image Processing**: Runs the full pipeline on each image
3. **Progress Tracking**: Shows progress and collects success/failure statistics
4. **Summary**: Provides a summary of processed images

### Result Comparison
1. **Result Collection**: Gathers processing results for multiple images
2. **Grid Generation**: Creates a visualization grid with aligned columns
3. **Visual Comparison**: Shows side-by-side comparisons of all processing steps
4. **Output**: Saves a single composite image with all comparisons
