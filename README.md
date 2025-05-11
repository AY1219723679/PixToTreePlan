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
- PIL (Pillow)
- scipy
- scikit-image
- Open3D (for point cloud generation)
- cv2 (OpenCV)

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

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained models:
   - For semantic segmentation (DeepLabV3+):
     - Download the model file from [this link](https://drive.google.com/file/d/1Bgs_5VBT_7F2NH9ObO_cs0J8CD_xJA0i/view)
     - Place it in the `checkpoints/` directory as `best_deeplabv3plus_mobilenet_cityscapes_os16.pth`
   - For depth estimation (MiDaS):
     - The script will automatically download the model if needed
     - Or you can manually download it from [GitHub releases](https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt)
     - Place it in the `midas_model/model-small.pt` path

## Usage

### Single Image Processing
Run the complete processing pipeline on a single image with:

```bash
python main.py --image_path input_images/your_image.jpg
```

### Batch Processing
Process multiple images in a directory with:

```bash
python batch_process.py --input_dir input_images --extensions jpg,png
```

### Pipeline Steps
The processing pipeline performs the following sequential operations:

1. **Ground Mask Generation**: Creates a semantic segmentation mask for ground surfaces using DeepLabV3+
2. **Cutout Creation**: Uses the ground mask to create a transparent cutout of the ground
3. **Depth Map Generation**: Uses MiDaS to generate a depth map from the cutout image
4. **Point Cloud Creation**: Combines the cutout and depth map to create a 3D point cloud

### Command Line Arguments

#### For Single Image Processing (`main.py`)
```
--image_path          Path to the input image
--resize_factor       Resize factor for segmentation (default: 0.6)
--min_region_size     Minimum size of regions to preserve in pixels (default: 400)
--min_area_ratio      Min component size as percentage of image area (default: 0.1)
--z_scale             Scale factor for Z values in point cloud (default: 0.5)
--sample_rate         Sample rate for point cloud generation (default: 2)
--visualize           Generate visualizations for each step
```

#### For Batch Processing (`batch_process.py`)
```
--input_dir           Directory containing input images (default: input_images)
--extensions          File extensions to process, comma-separated (default: jpg,jpeg,png)
--resize_factor       Resize factor for segmentation (default: 0.6)
--min_region_size     Minimum size of regions to preserve in pixels (default: 400)
--min_area_ratio      Min component size as percentage of image area (default: 0.1)
--z_scale             Scale factor for Z values in point cloud (default: 0.5)
--sample_rate         Sample rate for point cloud generation (default: 2)
--visualize           Generate visualizations for each step
```

### Example Usage
Standard single image processing:
```bash
python main.py --image_path input_images/urban_tree_33.jpg
```

High-detail processing with visualization:
```bash
python main.py --image_path input_images/urban_tree_33.jpg --resize_factor 0.8 --min_region_size 200 --z_scale 1.0 --sample_rate 1 --visualize
```

Process all JPG images in a directory:
```bash
python batch_process.py --input_dir my_images --extensions jpg --visualize
```

### Quick Start Scripts
For Windows users, we provide convenience scripts to run batch processing:

Using PowerShell:
```
.\run_batch_process.ps1
```

Using Command Prompt:
```
run_batch_process.bat
```

These scripts will:
1. Process all images in the default input directory
2. Generate visualizations for each step
3. Create a comparison grid showing all processed images side by side

## Component Removal Options
For cleaning up the ground mask:
- **Standard cleaning**: removes components < 10% of image area
- **Aggressive cleaning**: removes components < 15% of image area
- **Custom cleaning**: adjust parameters based on your specific needs

## Output Files and Directories
The pipeline generates the following outputs:

### For Single Image Processing:
- `ground_mask.png`: Binary ground mask from segmentation
- `cutout_ground.png`: Transparent cutout of the ground area
- `output_depth/`: Contains depth map files
  - `depth_map.png`: Complete depth map
  - `depth_masked.png`: Depth map with background removed
  - `depth_visualization.png`: Visualization of depth results
- `output_groundmasks/`: Contains ground mask files with original image names
  - `{filename}_groundmask.png`: Copy of the ground mask
  - `{filename}_cutout.png`: Copy of the cutout image
- `output_pointcloud/`: Contains 3D point cloud files
  - `{filename}_pointcloud.ply`: 3D point cloud in PLY format
  - `{filename}_visualization.png`: Visualization of the point cloud (when using `--visualize`)

### For Batch Processing:
The same directory structure is used, with each image's outputs named according to the original image filename.
The batch processor will also display a summary of successfully processed images and any failures.

## License
The code in this repository is available under the MIT License.
