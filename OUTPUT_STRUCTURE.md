# Output Folder Structure

## New Organization

PixToTreePlan now organizes all output files for each image in a dedicated subfolder, making it easy to manage and compare results.

## Folder Structure

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

## Legacy Support

For backward compatibility, the system also maintains the previous output folder structure:

```
output_groundmasks/
  {image_name}_groundmask.png
  {image_name}_cutout.png
  ...
output_depth/
  depth_map.png
  depth_masked.png
  ...
output_pointcloud/
  {image_name}_pointcloud.ply
  ...
```

## Advantages

1. **Organization**: All outputs for a single image are grouped together
2. **Clean Structure**: Standard naming conventions make scripts more maintainable
3. **Easier Comparison**: Better structure for comparing results between images
4. **No Overwrites**: Each image gets its own folder, preventing file conflicts

## Usage

The new structure is automatically used when processing images. No changes needed to your workflow.

- Single image processing: `python main.py --image_path=your_image.jpg`
- Batch processing: `python batch_process.py --input_dir=your_images_folder`

## Accessing Results

After processing, results are available in the `outputs/{image_name}` directory.
