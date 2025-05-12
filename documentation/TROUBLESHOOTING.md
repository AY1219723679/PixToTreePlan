# PixToTreePlan Troubleshooting Guide

This document provides solutions to common issues you might encounter when running the PixToTreePlan project with the new directory structure.

## Import Errors

If you encounter import errors like:
```
Error importing ground mask modules: No module named 'main.get_ground_mask'; 'main' is not a package
```

### Solution 1: Create symbolic link (Recommended)

The easiest solution is to create a symbolic link in the `core` directory pointing to the `main` module:

#### Windows
```
cd path\to\PixToTreePlan
mklink /J core\main main
```

#### macOS/Linux
```
cd path/to/PixToTreePlan
ln -sf ../main core/main
```

The batch scripts and VS Code tasks have been updated to do this automatically.

### Solution 2: Use the wrapper scripts

You can use the provided wrapper scripts in the project root which handle the imports correctly:
- `run_batch.py`: For batch processing of images
- `run_single.py`: For processing a single image

Example:
```
python run_batch.py --input_dir "dataset/input_images" --max_images 3 --visualize
```

### Solution 3: Set PYTHONPATH environment variable

Before running the scripts, set the PYTHONPATH environment variable to include the project root:

#### Windows
```
set PYTHONPATH=path\to\PixToTreePlan
python core/batch_process.py [arguments]
```

#### macOS/Linux
```
PYTHONPATH=path/to/PixToTreePlan python core/batch_process.py [arguments]
```

## File Path Errors

If you encounter errors related to file paths, especially with non-ASCII characters:

### Solution: Use absolute paths and ASCII filenames

- Use absolute paths when specifying input files
- Avoid using non-ASCII characters in file paths or filenames
- If using paths with spaces, enclose them in quotes

## Common OpenCV Errors

If you encounter OpenCV errors like:
```
OpenCV(4.11.0) error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
```

### Solution:
- Make sure the image files exist and are valid images
- Try using a temporary directory without non-ASCII characters for processing
- Convert image paths to absolute paths before processing

## No Valid Checkpoint Found

If you see the warning:
```
No valid checkpoint found in any of the expected locations
```

### Solution:
Make sure the checkpoint files exist in the `checkpoints` directory:
- `best_deeplabv3_resnet101_cityscapes.pth`
- `best_deeplabv3plus_mobilenet_cityscapes_os16.pth`

If they don't exist, you may need to download them separately.

## Other Issues

For other issues:
1. Check that all required Python packages are installed using `pip install -r requirements.txt`
2. Make sure your image paths don't contain non-ASCII characters
3. Try processing a small number of images first (`--max_images 1`) to identify any issues
4. Check the output directory to see if partial results were generated

If problems persist, please open an issue on the GitHub repository with detailed error information.
