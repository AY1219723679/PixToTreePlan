# PixToTreePlan

## Project Overview
PixToTreePlan is an image segmentation tool that extracts ground masks from images using the DeepLabV3+ model with a MobileNet backbone trained on the Cityscapes dataset. It's designed to identify ground-related classes (roads, sidewalks, terrain) and generate clean binary masks that can be used for further analysis or visualization.

## Features
- **Advanced Segmentation**: Uses DeepLabV3+ with MobileNet backbone for accurate ground surface detection
- **Component Removal**: Automatically filters out small disconnected regions in the ground mask
- **Customizable Parameters**: Fine-tune the segmentation and cleaning process to meet specific needs
- **Visualization Tools**: Generate detailed visualizations with class labels and component analysis

## Requirements
- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- PIL
- scipy
- scikit-image

## Installation
1. Clone this repository:
```bash
git clone https://github.com/AY1219723679/PixToTreePlan.git
cd PixToTreePlan
```

2. Download the DeepLabV3Plus-Pytorch implementation:
```bash
git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git
```

3. Download the pre-trained model:
- Download the model file from [this link](https://drive.google.com/file/d/1Bgs_5VBT_7F2NH9ObO_cs0J8CD_xJA0i/view) 
- Place it in the `checkpoints/` directory

## Usage
Run the ground mask extraction script with:

```bash
python single_image_processing_final.py
```

### Adjustable Parameters
The script provides several parameters to customize the ground mask extraction:

1. **resize_factor** (default: 0.6)
   - Controls the coarseness of segmentation
   - Lower values (0.3-0.5) produce smoother results with less detail
   - Higher values (0.7-1.0) preserve more detail but may introduce noise

2. **min_region_size** (default: 400)
   - Minimum size of regions (in pixels) to preserve during initial segmentation
   - Higher values remove more small isolated regions

3. **min_area_ratio** (default: 0.1 = 10%)
   - Components smaller than this percentage of the image area will be removed
   - Increase to remove larger disconnected regions

4. **relative_size_threshold** (default: 0.0 = disabled)
   - Secondary threshold as a ratio of the largest component size
   - Set > 0 to preserve components based on relative size

## Component Removal Options
For cleaning up the ground mask:
- **Standard cleaning**: removes components < 10% of image area
- **Aggressive cleaning**: removes components < 15% of image area
- **Custom cleaning**: adjust parameters based on your specific needs

## Output Files
- `segmentation_results.png`: Shows original image, class segmentation, and final ground mask
- `component_removal_comparison.png`: Visual comparison of original and cleaned masks
- `ground_mask.png`: Final binary ground mask (white = ground)

## License
The code in this repository is available under the MIT License.
