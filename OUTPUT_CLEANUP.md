# Output File Location Standardization

The PixToTreePlan codebase has been updated to standardize all output file locations. All image outputs (segmentation maps, ground masks, depth maps, point cloud visualizations, etc.) are now saved exclusively in the structured `outputs/` directory.

## Changes Made

1. **Removed Legacy File Locations**: Files are no longer saved to:
   - Root directory (e.g., `ground_mask.png`, `cutout_ground.png`)
   - Legacy output directories (`output_depth/`, `output_groundmasks/`, `output_pointcloud/`)

2. **Standardized Output Structure**: All outputs for an image are now saved in the image's dedicated subfolder:
   ```
   outputs/
     image_name/
       original.png
       segmentation.png
       ground_mask.png
       cutout.png
       depth_map.png
       depth_masked.png
       depth_visualization.png
       point_cloud.ply
       point_cloud_visualization.png
   ```

3. **Temporary File Management**: Temporary files created during processing are now:
   - Created in the image's output directory (not in the root)
   - Cleaned up automatically after processing

## Benefits

1. **Cleaner Workspace**: No more cluttering the root directory with temporary image files
2. **Better Organization**: All outputs for a specific image are grouped together
3. **Easier Management**: Find all processing results for an image in one place
4. **Consistent Structure**: Standardized naming across all processing stages

## Cleanup

A cleanup script (`cleanup_files.py`) has been provided to remove any leftover files from previous runs.
Run this to clean up:

```bash
python cleanup_files.py
```

## Technical Details

Files that previously saved to multiple locations:

1. `ground_mask.png`: Previously saved in both root and image output directories
2. `cutout_ground.png`: Previously saved in root directory
3. Depth maps: Previously saved in `output_depth/` directory
4. Point cloud files: Previously saved in `output_pointcloud/` directory

All file saving operations now exclusively target the `outputs/image_name/` directory.
