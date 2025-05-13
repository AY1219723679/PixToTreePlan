# 3D Point Cloud and YOLO Centers Visualization

This directory contains tools for visualizing:
1. Ground point clouds from PLY files
2. 3D center points derived from YOLO bounding boxes

## Overview

These scripts provide interactive 3D visualizations that combine point clouds with object center points. This is useful for relating detected objects (from YOLO) with their positions in a 3D ground reconstruction.

## Quick Start

### Prerequisites
Install required packages:
```
pip install -r visualization_requirements.txt
```

### Run the Example Visualization
```
python run_yolo_centers_3d.py
```

This will generate an HTML file (`yolo_centers_3d.html`) with an interactive 3D visualization.

## Available Scripts

### `visualize_combined_pointclouds.py`
The main visualization tool that can:
- Load point clouds from PLY files
- Load YOLO bounding boxes and project their centers to 3D using a depth map
- Create interactive 3D visualizations with Plotly

Usage:
```
python visualize_combined_pointclouds.py --ply=<path_to_ply> [--image=<path_to_image>] [--label=<path_to_label>] [--depth=<path_to_depth>] [--output=<output.html>]
```

### `run_yolo_centers_3d.py`
A simplified wrapper that runs the visualization with example data:
```
python run_yolo_centers_3d.py [--output=<output.html>]
```

### `yolo_centers_to_3d.py`
A specialized tool focused on projecting YOLO box centers to 3D:
```
python yolo_centers_to_3d.py --image=<path_to_image> --label=<path_to_label> --depth=<path_to_depth> --ply=<path_to_ply> [--output=<output.html>]
```

## Integration with Batch Processing

To visualize point clouds generated during batch processing:

1. Run the batch processing script:
```
python run_batch.py --input_dir=dataset/input_images --visualize
```

2. Once complete, run the visualization tool pointing to the generated PLY files:
```
python visualize_combined_pointclouds.py --ply=outputs/<output_folder>/point_cloud.ply
```

## Features

- **Interactive 3D Visualization**: Rotate, pan, zoom, and explore the point cloud
- **Combined View**: See YOLO-detected object centers alongside the ground point cloud
- **Vertical Guide Lines**: Lines connecting object centers to the ground for better orientation
- **Color Preservation**: Point cloud colors from the PLY file are maintained
- **Save as HTML**: Interactive visualizations can be saved as standalone HTML files

## Customization

The visualization parameters can be adjusted:
- `--point_size`: Size of point cloud points (default: 2.0)
- `--center_size`: Size of center point markers (default: 10.0)
- `--z_scale`: Scale factor for depth values (default: 0.5)
- `--title`: Title for the plot
