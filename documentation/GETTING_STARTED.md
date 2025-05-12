# Getting Started with PixToTreePlan

## Quick Start Guide

1. **Setup Environment**
   ```bash
   # Clone the repository
   git clone https://github.com/AY1219723679/PixToTreePlan.git
   cd PixToTreePlan
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Download DeepLabV3+ implementation (if not already included)
   git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git
   ```

2. **Download Pre-trained Models**
   - DeepLabV3+ model for semantic segmentation:
     - Download from [Google Drive](https://drive.google.com/file/d/1Bgs_5VBT_7F2NH9ObO_cs0J8CD_xJA0i/view)
     - Place in `checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth`
   
   - MiDaS model for depth estimation will download automatically on first run.

3. **Run the Pipeline**
   ```bash
   # Process a default image
   python main.py
   
   # Process your own image with default parameters
   python main.py --image_path path/to/your/image.jpg
   
   # Process with custom parameters
   python main.py --image_path path/to/your/image.jpg --resize_factor 0.7 --min_region_size 300 --visualize
   ```

## Understanding the Results

After running the pipeline, several output files and directories will be created:

1. **Ground Masks** (`output_groundmasks/`)
   - Binary masks highlighting ground surfaces
   - Transparent cutouts of the ground areas
   - Visualizations of segmentation results

2. **Depth Maps** (`output_depth/`)
   - Depth estimation visualizations
   - Masked depth maps (ground only)
   - Full scene depth maps

3. **Point Clouds** (`output_pointcloud/`)
   - 3D point clouds in PLY format
   - Visualizations (when `--visualize` is enabled)

## Customizing the Process

You can adjust various parameters to fine-tune the results:

- **Segmentation Quality**: Adjust `--resize_factor` between 0.3 (coarse) and 1.0 (detailed)
- **Region Cleaning**: Increase `--min_region_size` to remove more small isolated regions
- **Component Size**: Adjust `--min_area_ratio` to control minimum component size for retention
- **Point Cloud Density**: Lower `--sample_rate` for denser point clouds
- **Depth Scale**: Adjust `--z_scale` to control height variation in the point cloud

## Visualizing Point Clouds

To view the generated point clouds:
1. Use the built-in visualizer with the `--visualize` flag
2. Open the PLY files in software like MeshLab, CloudCompare, or Blender

## Troubleshooting

- **Missing Dependencies**: Ensure all requirements are installed
- **CUDA Errors**: If you encounter CUDA out-of-memory errors, reduce the `resize_factor` parameter
- **Point Cloud Quality**: For better point clouds, try:
  - Using images with good lighting and contrast
  - Increasing `z_scale` if terrain appears too flat
  - Decreasing `sample_rate` for more detail

For more detailed information, refer to the [README.md](README.md) and [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) files.
