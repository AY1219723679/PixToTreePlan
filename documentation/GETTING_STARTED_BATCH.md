# Getting Started with Batch Processing in PixToTreePlan

This guide explains how to use the batch processing functionality in PixToTreePlan to process multiple images at once.

## Prerequisites

Before you begin, make sure you've:

1. Installed all required dependencies from `requirements.txt`
2. Downloaded the necessary model files as described in the main README.md
3. Prepared a directory containing the images you want to process

## Quick Start (Windows)

For Windows users, we provide convenience scripts that handle everything for you:

Using PowerShell:
```
.\run_batch_process.ps1
```

Using Command Prompt:
```
run_batch_process.bat
```

These scripts will automatically:
- Process all images in the dataset/input_images directory
- Generate visualizations for each processing step
- Create a comparison grid showing all results side by side

## Basic Batch Processing

If you prefer more control, you can run the batch processor directly:

```bash
python batch_process.py --input_dir input_images
```

This will:
- Look for all JPG, JPEG, and PNG files in the `input_images` directory
- Process each image through the full pipeline
- Save outputs in their respective directories with filenames based on the original image names

## Customizing Batch Processing

You can customize the batch processing with various command-line options:

```bash
python batch_process.py --input_dir my_images --extensions tif,jpg --resize_factor 0.8 --min_region_size 200 --visualize
```

### Common Options Explained

- `--input_dir`: Directory containing your input images
- `--extensions`: Comma-separated list of file extensions to process (e.g., "jpg,png,tif")
- `--resize_factor`: Resize factor for semantic segmentation (lower values are faster but less detailed):
  - 0.3 = Fast, lower quality
  - 0.6 = Balanced (default)
  - 1.0 = Highest quality, slowest
- `--min_region_size`: Minimum size (in pixels) of regions to keep in the mask:
  - Lower values (e.g., 200) keep more small details
  - Higher values (e.g., 600) remove small disconnected regions
- `--min_area_ratio`: Minimum component size as a ratio of image area:
  - Lower values (e.g., 0.05) keep smaller components
  - Higher values (e.g., 0.2) keep only large components
- `--z_scale`: Scale factor for depth in point clouds:
  - Lower values (e.g., 0.3) create flatter point clouds
  - Higher values (e.g., 1.0) emphasize height differences
- `--visualize`: Generate visualization images for each processing step

## Processing Workflow

For each image in the input directory, the batch processor will:

1. **Generate a Ground Mask** using DeepLabV3+ semantic segmentation
   - Identifies ground-related classes (roads, sidewalks, terrain)
   - Removes small disconnected regions
   - Saves the mask as `output_groundmasks/{filename}_groundmask.png`

2. **Create a Cutout** using the ground mask
   - Isolates the ground with transparency
   - Saves the cutout as `output_groundmasks/{filename}_cutout.png`

3. **Generate a Depth Map** using MiDaS
   - Creates a depth estimation of the ground surface
   - Saves both masked and unmasked depth maps to `output_depth/`

4. **Create a 3D Point Cloud** (if Open3D is available)
   - Combines the cutout and depth map to create a 3D representation
   - Saves the point cloud as `output_pointcloud/{filename}_pointcloud.ply`
   - Optionally saves a visualization as `output_pointcloud/{filename}_visualization.png`

## Tips for Batch Processing

1. **Process a subset first**: Test with a few images before processing a large directory
2. **Check memory usage**: Processing high-resolution images can be memory-intensive
3. **Adjust parameters**: Different types of images may benefit from different settings
4. **Check for failures**: The script will provide a summary of successful and failed images

## Troubleshooting

### Common Issues

1. **Out of Memory**: 
   - Use a lower `resize_factor` (e.g., 0.4 instead of 0.6)
   - Process fewer images at once

2. **Poor Quality Results**:
   - Try increasing `resize_factor` to 0.8 or higher
   - Adjust `min_region_size` based on your image resolution

3. **Missing Point Clouds**:
   - Verify that Open3D is properly installed
   - Check error messages for specific issues

4. **Long Processing Time**:
   - Reduce the `resize_factor` for faster processing
   - Increase the `sample_rate` for point cloud generation
   - Process fewer images at once

5. **File Not Found Errors**:
   - The batch processor now uses a wrapper script to handle file paths correctly
   - If you see "FileNotFoundError" messages, make sure you're running the batch_process.py script from the project root directory
   - For manual processing, you can use the wrapper.py script directly:
     ```
     python wrapper.py --image_path=/path/to/your/image.jpg
     ```

For more detailed information, refer to the main README.md file.
