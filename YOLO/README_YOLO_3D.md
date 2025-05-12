# YOLO 3D Comparison Tool

This tool helps convert YOLO bounding boxes to 3D points using depth maps. It includes utilities to ensure depth maps are correctly matched to images.

## Recent Fixes

We resolved an issue where the depth map and masked depth map could be loaded from different folders, causing mismatches. The solution:

1. Improved the matching algorithm to accurately find the correct output folder for each image
2. Added validation to ensure depth maps match image dimensions
3. Created a diagnostic tool `check_depth_maps.py` to identify and fix mismatches
4. Added command-line flags to control validation and automatic ground mask generation

## Usage

### Check and Run (Recommended)

The easiest way to use the tool is with the `check_and_run` script, which automatically checks for matching depth maps and runs the comparison:

**PowerShell:**
```powershell
.\check_and_run.ps1 -image "path\to\image.jpg" [-auto_ground] [-force]
```

**Batch:**
```
check_and_run.bat path\to\image.jpg [--auto_ground] [--force]
```

### Manual Usage

To run the tool manually:

```
python yolo_3d_compare.py --image <path_to_image> --label <path_to_label> --depth <path_to_depth> [--depth_masked <path_to_masked_depth>]
```

#### Optional Arguments

- `--auto_ground`: Automatically generate a ground-only depth map if not provided
- `--skip_validation`: Skip validation of depth maps against image dimensions
- `--force`: Continue even if validation fails (no prompts)
- `--z_scale`: Scale factor for depth values (default: 0.5)
- `--output_dir`: Directory to save output files (default: yolo_3d_output)

### Diagnostic Tool

To check if an image has correct matching depth maps:

```
python check_depth_maps.py --image <path_to_image> [--fix]
```

## Output

The tool generates:
- 3D point clouds from YOLO bounding boxes using different depth maps
- Visualizations comparing the 3D points from regular and ground-only depth maps
- PLY files that can be imported into 3D software

## Troubleshooting

If you experience issues with depth maps not matching images:

1. Run the diagnostic tool: `python check_depth_maps.py --image <path_to_image>`
2. Check the output folders to verify that depth maps exist
3. Use the `--auto_ground` flag if you don't have ground-only depth maps
4. Use the `--force` flag to override dimension validation if needed

## Dependencies

- Python 3.7+
- NumPy
- Matplotlib
- PIL/Pillow
- OpenCV (cv2)
