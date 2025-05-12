# YOLO Bounding Box Utilities

This documentation covers the utilities for working with YOLO format bounding boxes in the PixToTreePlan project.

## YOLO Format Explanation

YOLO (You Only Look Once) uses a specific format for bounding box annotations:

```
[class_id] [x_center_norm] [y_center_norm] [width_norm] [height_norm]
```

Where:
- `class_id`: Integer representing the object class (0-based index)
- `x_center_norm`: X-coordinate of box center, normalized between 0 and 1
- `y_center_norm`: Y-coordinate of box center, normalized between 0 and 1
- `width_norm`: Width of the box, normalized between 0 and 1
- `height_norm`: Height of the box, normalized between 0 and 1

All coordinates are relative to the image dimensions:
- For an 800×600 image, a box with coordinates [0.5, 0.5, 0.25, 0.25] would be centered at (400, 300) with a width of 200 and height of 150 pixels.

## Utility Functions

The `yolo_utils.py` module provides functions for working with YOLO format bounding box annotations:

### `load_yolo_bboxes(label_path, image_path=None, image_size=None)`

Loads YOLO format bounding boxes and converts them to pixel coordinates.

**Parameters:**
- `label_path`: Path to the YOLO label file (.txt)
- `image_path`: Path to the corresponding image (to get dimensions)
- `image_size`: Tuple of (width, height) if image_path is not provided

**Returns:**
A list of dictionaries containing:
```python
{
    'class_id': int,
    'x_center': float,  # in pixels
    'y_center': float,  # in pixels
    'width': float,     # in pixels
    'height': float,    # in pixels
    'x1': float,        # top-left x
    'y1': float,        # top-left y
    'x2': float,        # bottom-right x
    'y2': float         # bottom-right y
}
```

### `load_yolo_dataset(labels_dir, images_dir=None)`

Loads an entire YOLO dataset, converting all labels to pixel coordinates.

**Parameters:**
- `labels_dir`: Directory containing label files
- `images_dir`: Directory containing corresponding images

**Returns:**
A dictionary mapping filename (without extension) to a list of bounding boxes.

### `visualize_bboxes(image_path, bboxes, output_path=None, color=(255, 0, 0), thickness=2)`

Visualizes bounding boxes on an image.

**Parameters:**
- `image_path`: Path to the image
- `bboxes`: List of bounding box dictionaries (from load_yolo_bboxes)
- `output_path`: Path to save the output image (optional)
- `color`: RGB color for drawing the boxes
- `thickness`: Line thickness for the boxes

**Returns:**
A PIL Image object with the drawn bounding boxes.

## Example Usage

```python
from yolo_utils import load_yolo_bboxes, visualize_bboxes

# Load and convert bounding boxes
label_path = "YOLO/train/labels/urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.txt"
image_path = "YOLO/train/images/urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg"

bboxes = load_yolo_bboxes(label_path, image_path)

# Print some information
for i, bbox in enumerate(bboxes[:3]):
    print(f"Box {i+1}: Class {bbox['class_id']}, "
          f"Center: ({bbox['x_center']:.1f}, {bbox['y_center']:.1f}), "
          f"Size: {bbox['width']:.1f}x{bbox['height']:.1f}")

# Visualize the boxes
visualize_bboxes(image_path, bboxes, "output_visualization.jpg")
```

## Integration with PixToTreePlan

These utilities can be used to incorporate bounding box data from YOLO models into the PixToTreePlan pipeline, enabling:

1. Object detection for specific elements in tree scenes
2. Automated region of interest selection
3. Integration of tree trunk detection with depth estimation
4. Enhanced visualization of detected objects alongside 3D point clouds

## Notes

- Class IDs should correspond to the classes defined in the `data.yaml` file
- The current implementation in this project defines class 0 as "trunk" for tree trunk detection
