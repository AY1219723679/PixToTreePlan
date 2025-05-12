import os
import numpy as np
from PIL import Image


def load_yolo_bboxes(label_path, image_path=None, image_size=None):
    """
    Load YOLO format bounding boxes from a label file and convert normalized coordinates
    to pixel coordinates.
    
    YOLO format is: [class_id, x_center_norm, y_center_norm, width_norm, height_norm]
    where all values except class_id are normalized between 0 and 1.
    
    Args:
        label_path (str): Path to the YOLO label file (.txt)
        image_path (str, optional): Path to the corresponding image file to get dimensions.
            If None, image_size must be provided.
        image_size (tuple, optional): Tuple of (width, height) if image_path is not provided.
            If None, image_path must be provided.
            
    Returns:
        list: List of dictionaries containing:
            {
                'class_id': int,
                'x_center': float (in pixels),
                'y_center': float (in pixels),
                'width': float (in pixels),
                'height': float (in pixels),
                'x1': float (top-left x),
                'y1': float (top-left y),
                'x2': float (bottom-right x),
                'y2': float (bottom-right y)
            }
    
    Raises:
        ValueError: If neither image_path nor image_size is provided.
    """
    # Check if we have a way to get image dimensions
    if image_path is None and image_size is None:
        raise ValueError("Either image_path or image_size must be provided")
    
    # Get image dimensions
    if image_size is not None:
        img_width, img_height = image_size
    else:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    
    bboxes = []
    
    # Read bounding box data from label file
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse YOLO format: class_id x_center y_center width height
            parts = line.split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
            
            # Convert normalized coordinates to pixel values
            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            width = width_norm * img_width
            height = height_norm * img_height
            
            # Calculate corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            bbox = {
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            }
            
            bboxes.append(bbox)
    
    return bboxes


def load_yolo_dataset(labels_dir, images_dir=None):
    """
    Load an entire YOLO dataset, converting all labels to pixel coordinates.
    
    Args:
        labels_dir (str): Directory containing label files
        images_dir (str, optional): Directory containing corresponding images.
            If not provided, image dimensions can't be used and the function will fail.
            
    Returns:
        dict: Dictionary mapping filename (without extension) to list of bounding boxes
    """
    result = {}
    
    # Iterate through all label files in the directory
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
            
        base_name = os.path.splitext(label_file)[0]
        label_path = os.path.join(labels_dir, label_file)
        
        # Try to find corresponding image file
        image_path = None
        if images_dir is not None:
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
        
        if image_path is None:
            continue
            
        # Load bounding boxes
        bboxes = load_yolo_bboxes(label_path, image_path)
        result[base_name] = bboxes
    
    return result


def visualize_bboxes(image_path, bboxes, output_path=None, color=(255, 0, 0), thickness=2):
    """
    Visualize bounding boxes on an image.
    
    Args:
        image_path (str): Path to the image
        bboxes (list): List of bounding box dictionaries (from load_yolo_bboxes)
        output_path (str, optional): Path to save the output image. If None, will display.
        color (tuple): RGB color for drawing the boxes
        thickness (int): Line thickness
        
    Returns:
        PIL.Image: The image with drawn bounding boxes
    """
    try:
        import cv2
        
        # Load image with OpenCV
        img = cv2.imread(image_path)
        
        # Draw each bounding box
        for bbox in bboxes:
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Convert RGB color to BGR for OpenCV
            bgr_color = (color[2], color[1], color[0])
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, thickness)
            
            # Optionally draw class label
            class_id = bbox['class_id']
            label = f"Class {class_id}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        
        # Save or return the image
        if output_path:
            cv2.imwrite(output_path, img)
        
        # Convert back to PIL Image format for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        return Image.fromarray(img_rgb)
    
    except ImportError:
        print("OpenCV (cv2) is required for visualization. Install with: pip install opencv-python")
        # Fall back to PIL if OpenCV isn't available
        from PIL import Image, ImageDraw
        
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        for bbox in bboxes:
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=thickness)
        
        if output_path:
            img.save(output_path)
        
        return img


if __name__ == "__main__":
    """
    Example usage
    """
    # Example paths - update these to match your file structure
    labels_dir = "train/labels"
    images_dir = "train/images"
    
    # Process a single label file
    label_file = "urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.txt"
    image_file = "urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg"
    
    label_path = os.path.join(labels_dir, label_file)
    image_path = os.path.join(images_dir, image_file)
    
    if os.path.exists(label_path) and os.path.exists(image_path):
        # Load and convert bounding boxes
        bboxes = load_yolo_bboxes(label_path, image_path)
        
        # Display first few boxes
        print(f"Found {len(bboxes)} bounding boxes:")
        for i, bbox in enumerate(bboxes[:3]):  # Show first 3 boxes
            print(f"Box {i+1}: Class {bbox['class_id']}, "
                  f"Center: ({bbox['x_center']:.1f}, {bbox['y_center']:.1f}), "
                  f"Size: {bbox['width']:.1f}x{bbox['height']:.1f}")
        
        # Visualize (if OpenCV is installed)
        try:
            output_path = "bbox_visualization.jpg"
            visualize_bboxes(image_path, bboxes, output_path)
            print(f"Visualization saved to {output_path}")
        except Exception as e:
            print(f"Visualization failed: {e}")
    else:
        print("Example files not found. Update the paths in the script.")
