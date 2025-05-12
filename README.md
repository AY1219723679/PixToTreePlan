# PixToTreePlan

<div align="center">
  <img src="documentation/assets/header_image.png" alt="PixToTreePlan Header" width="800"/>
</div>

PixToTreePlan is an image processing pipeline that transforms 2D images of trees and landscapes into 3D point cloud representations. The tool uses semantic segmentation, depth estimation, and point cloud generation to create detailed 3D models from regular photographs.

## 🚀 Features

- **Ground Mask Extraction**: Identifies ground surfaces in images using DeepLabV3+ semantic segmentation
- **Depth Estimation**: Generates detailed depth maps using MiDaS deep learning model
- **3D Point Cloud Generation**: Creates colored point clouds from images and depth maps
- **Batch Processing**: Process multiple images at once with customizable parameters
- **Visualization Tools**: View results at each processing stage and compare outputs

## 📋 Requirements

- Python 3.7+
- PyTorch 1.7+
- OpenCV
- Open3D (for point cloud generation)
- Additional dependencies in `requirements.txt`

## 🔧 Installation

1. Clone this repository:
   ```powershell
   git clone https://github.com/AY1219723679/PixToTreePlan.git
   cd PixToTreePlan
   ```

2. Install the required packages:
   ```powershell
   pip install -r requirements.txt
   ```

3. Download required model checkpoints:
   - DeepLabV3+ checkpoints should be placed in the `checkpoints/` directory
   - MiDaS model should be placed in the `midas_model/` directory

## 🏃‍♂️ Quick Start

### Process a Batch of Images

Run the batch processing script to process all images in the input directory:

```powershell
.\process_batch.bat
```

Or use the Python wrapper directly:

```powershell
python run_batch.py --input_dir "dataset/input_images" --visualize
```

### Process a Single Image

```powershell
python run_single.py --image_path "dataset/input_images/urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg" --visualize
```

### Using VS Code Tasks

This project includes predefined VS Code tasks:

1. Press `Ctrl+Shift+P` and select **Tasks: Run Task**
2. Choose one of:
   - **Process Single Image**: Process a single image with visualization
   - **Batch Process Images**: Process multiple images from a directory
   - **Generate Comparison Grid**: Create a comparison visualization of results

## 📊 Pipeline Overview

<div align="center">
  <img src="documentation/assets/pipeline.png" alt="Processing Pipeline" width="800"/>
</div>

1. **Input Image**: Start with any photograph containing trees and ground
2. **Ground Mask**: Extract ground surfaces using semantic segmentation
3. **Depth Estimation**: Generate depth map for the masked ground area
4. **Point Cloud**: Convert image and depth to 3D point cloud
5. **Visualization**: View and analyze the results

## 📂 Project Structure

```
PixToTreePlan/
├── core/                  # Core processing scripts
│   ├── batch_process.py   # Batch processing script
│   ├── compare_results.py # Result comparison tool
│   └── main.py            # Main processing pipeline
├── dataset/               # Input image datasets
│   └── input_images/      # Default input images
├── main/                  # Processing modules
│   ├── generate_depth/    # Depth map generation
│   ├── get_ground_mask/   # Semantic segmentation
│   └── img_to_pointcloud/ # Point cloud generation
├── outputs/               # Output results
├── documentation/         # Documentation files
└── process_batch.bat      # Batch processing script
```

## ⚙️ Configuration Options

The processing pipeline can be customized with various parameters:

- `--resize_factor`: Resize factor for segmentation (default: 0.6)
- `--min_region_size`: Minimum size of regions to preserve in pixels (default: 400)
- `--min_area_ratio`: Minimum component size as percentage of image (default: 0.1)
- `--z_scale`: Scale factor for Z values in point cloud (default: 0.5)
- `--sample_rate`: Sample rate for point cloud generation (default: 2)
- `--visualize`: Generate visualizations for each step
- `--max_images`: Maximum number of images to process in batch mode

## 📄 Documentation

For more detailed information, see the documentation directory:

- [Getting Started](documentation/GETTING_STARTED.md)
- [Batch Processing Guide](documentation/BATCH_PROCESSING.md)
- [Output Structure](documentation/OUTPUT_STRUCTURE.md)
- [Project Structure](documentation/PROJECT_STRUCTURE.md)
- [Troubleshooting Guide](documentation/TROUBLESHOOTING.md)

## 📝 Recent Updates

- Fixed path handling for images outside the input_images folder
- Added batch size limiting with the `--max_images` parameter
- Improved project structure with core scripts in the `core/` directory
- Added automatic symbolic link setup for proper imports
- Created comprehensive troubleshooting documentation

## 📊 Results

<div align="center">
  <img src="documentation/assets/results.png" alt="Sample Results" width="800"/>
</div>

## ❓ Troubleshooting

If you encounter issues:

1. Check the [Troubleshooting Guide](documentation/TROUBLESHOOTING.md)
2. Ensure all dependencies are correctly installed
3. Verify that model checkpoints are in the correct locations
4. For import errors, make sure the symbolic link between `core/main` and `main` exists

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgements

- DeepLabV3+ for semantic segmentation
- MiDaS for monocular depth estimation
- Open3D for point cloud processing
