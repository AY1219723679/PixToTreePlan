# 3D Visualization Tools Documentation

This document explains how to use the 3D visualization tools for point clouds and YOLO object detection.

## Overview

The PixToTreePlan project includes several scripts for visualizing:
1. 3D point clouds from PLY files
2. YOLO object detection results projected into 3D space
3. Combined visualizations with proper coordinate alignment

## Quick Start

The easiest way to visualize your data is to use the "Ultimate" visualization script:

```bash
python run_ultimate_visualization.py
```

This will automatically find your point cloud, depth map, YOLO labels, and image files, and create an interactive 3D visualization.

## Available Scripts

### Ultimate Visualization (Recommended)

This script combines all the best features from our other visualization tools:

- **Script**: `ultimate_3d_visualization.py`
- **Runner**: `run_ultimate_visualization.py` 
- **Features**:
  - Excellent visibility of point cloud and object centers
  - Proper coordinate alignment between point cloud and image space
  - Guide lines connecting object centers to the ground
  - Color-coded object classes with labels
  - Reference coordinate axes
  - Interactive controls

```bash
# Basic usage:
python run_ultimate_visualization.py

# With options:
python run_ultimate_visualization.py --bright --point_size=8.0 --center_size=20.0

# Specify a different output folder:
python run_ultimate_visualization.py --folder "outputs/your_folder_name"

# For large point clouds, use subsampling for better performance:
python run_ultimate_visualization.py --subsample=0.2
```

### Maximum Visibility Point Cloud

For cases where you need to focus on the point cloud visibility:

- **Script**: `maximum_visibility_pointcloud.py`
- **Runner**: `run_max_visibility.py`
- **Features**:
  - Extra large points
  - High contrast colors
  - Optimized camera position

```bash
python run_max_visibility.py
```

### Coordinate Aligned Visualization

For ensuring proper alignment between point clouds and YOLO centers:

- **Script**: `coordinate_aligned_visualization.py`
- **Runner**: `run_aligned_visualization.py`
- **Features**: 
  - Ensures point cloud and center points are properly aligned
  - Shows coordinate reference axes

```bash
python run_aligned_visualization.py
```

## Script Parameters

### Ultimate 3D Visualization

```
python ultimate_3d_visualization.py --help

arguments:
  --ply PLY             Path to the point cloud PLY file
  --label LABEL         Path to the YOLO label file (.txt)
  --image IMAGE         Path to the original image (for YOLO parsing)
  --depth DEPTH         Path to the depth map file
  --output OUTPUT       Path to save the visualization HTML (default: ultimate_3d_viz.html)
  --point_size POINT_SIZE
                        Size of point cloud points (default: 5.0)
  --center_size CENTER_SIZE
                        Size of center points (default: 15.0)
  --bright_colors       Use bright colors for better visibility
  --subsample SUBSAMPLE
                        Subsample point cloud for performance (0.0-1.0)
  --class_names CLASS_NAMES
                        Path to class names file (one name per line)
```

## Data Requirements

For full functionality, you need:

1. **Point Cloud**: PLY file with 3D points and colors
2. **Original Image**: PNG/JPG file used for YOLO detection
3. **Depth Map**: PNG grayscale image where pixel intensity represents depth
4. **YOLO Labels**: Text file with YOLO format bounding boxes
5. **Class Names** (optional): Text file with class names, one per line

## Tips for Better Visualizations

1. **Visibility Issues**:
   - Use `--bright_colors` option for better visibility
   - Increase point size with `--point_size=10.0` for small point clouds
   - Try different camera angles by rotating the 3D view

2. **Performance Issues**:
   - For large point clouds, use subsampling: `--subsample=0.2`
   - Close other applications to free up memory

3. **Alignment Issues**:
   - Ensure depth maps correspond correctly to the original images
   - Check that YOLO labels are correctly generated
   - If points seem misaligned, try the coordinate alignment script first

## Troubleshooting

1. **"No point cloud found"**: Check the path to your PLY file
2. **"No valid 3D center points"**: Ensure your depth map and YOLO labels match
3. **"Browser hangs or crashes"**: Your point cloud may be too large; try subsampling
4. **"Points are not visible"**: Try increasing point size or using bright color mode
5. **"Center points are not aligned"**: Review the coordinate transformation in `project_center_points()`
