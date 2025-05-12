# Batch Processing Instructions

This document provides instructions on how to use the batch processing functionality in PixToTreePlan.

## Quick Start

The easiest way to process multiple images is to run one of the provided scripts:

- Windows Command Prompt: `process_all_images.bat`
- PowerShell: `.\process_all_images.ps1`

## Manual Execution

### Basic Usage

```
python batch_process.py --input_dir="dataset/input_images" --visualize
```

### Command Line Arguments

- `--input_dir`: Directory containing input images (default: 'input_images')
  - Recommended: Use "dataset/input_images" where most images are stored
- `--extensions`: File extensions to process (default: 'jpg,jpeg,png')
- `--resize_factor`: Resize factor for segmentation (default: 0.6, range: 0.3-1.0)
- `--min_region_size`: Minimum size of regions to preserve (default: 400 pixels)
- `--min_area_ratio`: Min component size as percentage of image area (default: 0.1 = 10%)
- `--z_scale`: Scale factor for Z values in point cloud (default: 0.5)
- `--sample_rate`: Sample rate for point cloud generation (default: 2, 1=full density)
- `--visualize`: Generate visualizations for each step
- `--max_images`: Maximum number of images to process

## Example Commands

Process 5 images with visualizations:
```
python batch_process.py --input_dir="dataset/input_images" --visualize --max_images=5
```

Change segmentation parameters:
```
python batch_process.py --input_dir="dataset/input_images" --resize_factor=0.5 --min_region_size=300
```

## Output Locations

- Ground masks: `output_groundmasks/`
- Depth maps: `output_depth/`
- Point clouds: `output_pointcloud/`

## Troubleshooting

If you encounter an error about missing images:
1. Check that images exist in one of these directories:
   - `input_images/`
   - `dataset/input_images/`
   - `dataset/get_test_imgs/`
2. Make sure the file extensions match the `--extensions` parameter
