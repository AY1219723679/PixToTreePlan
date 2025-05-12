"""
Visualization functions for ground mask and segmentation maps
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import os

def get_cityscapes_colormap():
    """
    Returns the Cityscapes dataset colormap for visualization
    """
    cityscapes_colormap = [
        [128, 64, 128],  # 0: Road
        [244, 35, 232],  # 1: Sidewalk
        [70, 70, 70],    # 2: Building
        [102, 102, 156], # 3: Wall
        [190, 153, 153], # 4: Fence
        [153, 153, 153], # 5: Pole
        [250, 170, 30],  # 6: Traffic Light
        [220, 220, 0],   # 7: Traffic Sign
        [107, 142, 35],  # 8: Vegetation
        [152, 251, 152], # 9: Terrain
        [70, 130, 180],  # 10: Sky
        [220, 20, 60],   # 11: Person
        [255, 0, 0],     # 12: Rider
        [0, 0, 142],     # 13: Car
        [0, 0, 70],      # 14: Truck
        [0, 60, 100],    # 15: Bus
        [0, 80, 100],    # 16: Train
        [0, 0, 230],     # 17: Motorcycle
        [119, 11, 32]    # 18: Bicycle
    ]
    return cityscapes_colormap

def apply_colormap(segmentation_map, colormap):
    """
    Applies a colormap to a segmentation map
    """
    colored_segmentation = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in enumerate(colormap):
        if class_id < len(colormap):  # Make sure we don't go out of bounds
            mask = segmentation_map == class_id
            colored_segmentation[mask] = color
    
    return colored_segmentation

def visualize_segmentation(segmentation_map, save_path=None):
    """
    Create a visualization of the segmentation map using the Cityscapes colormap
    
    Args:
        segmentation_map: The segmentation map with class IDs
        save_path: Path to save the visualization image
    """
    # Get colormap
    colormap = get_cityscapes_colormap()
    
    # Apply colormap to segmentation map
    colored_segmentation = apply_colormap(segmentation_map, colormap)
    
    # Save the visualization
    if save_path:
        Image.fromarray(colored_segmentation).save(save_path)
    
    return colored_segmentation
