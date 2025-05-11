# Output Structure Update

The PixToTreePlan codebase has been updated to exclusively save all image outputs in the designated `outputs/` directory. This change creates a cleaner, more organized workspace by removing any code that saves images outside of this folder.

## Changes Made

1. Updated `main.py` to save all outputs for each image in a dedicated subfolder
2. Modified depth generation to avoid saving temporary files outside output directory
3. Updated point cloud generation to respect the clean output structure
4. Removed all code that saved files to legacy directories
5. Added cleanup script to remove leftover files from previous runs

## New Folder Structure

```
outputs/
  image1_name/
    original.png         # Original input image
    segmentation.png     # Segmentation map visualization
    ground_mask.png      # Binary ground mask
    cutout.png           # Transparent cutout using the mask
    depth_map.png        # Full depth map
    depth_masked.png     # Depth map with ground mask applied
    point_cloud.ply      # 3D point cloud data
    point_cloud_visualization.png  # Point cloud visualization (if --visualize used)
  image2_name/
    ...
```

## Legacy Structures Removed

The system no longer saves files to the old output structure:

```
output_groundmasks/  (removed)
output_depth/        (removed)
output_pointcloud/   (removed)
```

All outputs are now saved exclusively in the organized `outputs/` directory.

## Benefits

- **Organization**: All outputs for a single image are grouped together
- **Clean Structure**: Standard naming conventions make scripts more maintainable
- **Easier Comparison**: Better structure for comparing results between images
- **No Overwrites**: Each image gets its own folder, preventing file conflicts

## How to Clean Up Legacy Files

Run the cleanup script to remove any leftover files from previous runs:

```bash
python cleanup_files.py
```

This will remove all legacy output directories while preserving the current output structure.

## Testing the Output Structure

To test the structure with a single image:

```bash
python main.py --image_path=dataset/input_images/urban_tree_10_jpg.rf.81eafaad33fd7ce2b7233b8483800d71.jpg
```

To test batch processing with the structure:

```bash
python batch_process.py --max_images=2
```

Both commands will now properly save all outputs to the `outputs/` directory.

## Technical Implementation

The changes focus on:

1. Creating a new `setup_directories()` function that sets up image-specific output directories
2. Saving all outputs to the image's output directory with standardized names
3. Copying results to legacy locations for backward compatibility
4. Adding proper error handling and reporting
