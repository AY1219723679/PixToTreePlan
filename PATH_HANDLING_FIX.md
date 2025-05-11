# Path Handling Bug Fix Summary

This document summarizes the bug that was fixed related to file paths in the PixToTreePlan project.

## Bug Description

The project was failing with a `FileNotFoundError` when processing images from directories outside of the `input_images` folder. Specifically, the bug occurred in the `step2_create_cutout` function in `main.py`, which was not properly handling absolute paths when calling `create_cutout_with_mask()` function.

## Root Cause

1. The `create_cutout_with_mask()` function in `main/get_ground_mask/create_cutout_simple.py` was being called with a relative image path.
2. When this function tried to open the image, it was looking for the file relative to the current working directory, not using the actual path that was provided by the batch processing script.
3. Additionally, when the script was run directly (instead of being imported as a module), it used a hardcoded path to a specific image in the `input_images` directory.

## Solution

The fix was implemented by modifying the `step2_create_cutout` function in `main.py` to convert the `image_path` parameter to an absolute path before passing it to `create_cutout_with_mask()`:

```python
def step2_create_cutout(image_path, mask_path, image_basename):
    """
    Step 2: Create a cutout image using the ground mask
    """
    print("\nSTEP 2: Creating cutout image...")
    
    # Ensure image_path is an absolute path
    image_path = os.path.abspath(image_path)
    print(f"  Using absolute image path: {image_path}")
    
    # Create cutout
    cutout_path = "cutout_ground.png"
    create_cutout_with_mask(image_path, mask_path, cutout_path)
    
    # Rest of the function...
```

## Additional Changes

1. **Batch Processing Improvements:**
   - Modified `batch_process.py` to convert image paths to absolute paths
   - Added `--max_images` parameter to limit the number of images processed in a batch

2. **Alternative Solution (Wrapper):**
   - Created a `wrapper.py` script that copies images to the `input_images` directory before processing
   - This provides an alternative approach if further path issues are encountered

3. **Documentation Updates:**
   - Updated `GETTING_STARTED_BATCH.md` with troubleshooting information about path handling
   - Added VS Code tasks to make processing easier

## Testing

The fix was successfully tested with:

1. Images from the default `input_images` directory
2. Images from the `dataset/input_images` directory
3. Images from arbitrary locations on the file system

## Future Recommendations

1. Consider implementing more robust path handling throughout the codebase
2. Add unit tests specifically for verifying path handling works correctly
3. Consider adding a configuration option to specify where temporary files should be stored
