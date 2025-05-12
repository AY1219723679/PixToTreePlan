# filepath: c:\Users\Ay121\OneDrive - Harvard University\MIT\Creative ML\single_image_processing.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from skimage import measure, morphology
import sys
import os

# Add DeepLabV3Plus-Pytorch to path
deeplab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepLabV3Plus-Pytorch")
sys.path.insert(0, deeplab_path)  # Insert at beginning to ensure our module takes precedence

# Specify the path to your image
IMAGE_PATH = "dataset/input_images/urban_tree_1_jpg.rf.3b0d0591cbe20a93bf8d9fb59761f634.jpg"  # Replace with your image path

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_model():
    """
    Load the DeepLabV3+ model with MobileNet backbone trained on Cityscapes
    """
    try:
        # First check if DeepLabV3Plus-Pytorch repository is available
        deeplab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepLabV3Plus-Pytorch")
        if not os.path.isdir(deeplab_path) or len(os.listdir(deeplab_path)) == 0:
            raise ImportError(f"DeepLabV3Plus-Pytorch directory is empty or missing at {deeplab_path}")
        
        # Import the DeepLabV3+ model
        from network.modeling import deeplabv3plus_mobilenet
        
        # Create the model with correct parameters for Cityscapes
        print("Creating DeepLabV3+ model with MobileNet backbone for Cityscapes (19 classes)...")
        model = deeplabv3plus_mobilenet(num_classes=19, output_stride=16)
        
        # Check for the checkpoint at multiple possible locations
        checkpoint_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                        "checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth")
        ]
        
        checkpoint_loaded = False
        for checkpoint_path in checkpoint_paths:
            if os.path.isfile(checkpoint_path):
                try:
                    # Attempt to load the checkpoint with proper error handling for PyTorch 2.6+
                    try:
                        # In PyTorch 2.6+, we need to explicitly set weights_only=False
                        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    except TypeError:
                        # For older PyTorch versions that don't have the weights_only parameter
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # Check if the checkpoint has the expected format
                    if "model_state" in checkpoint:
                        model.load_state_dict(checkpoint["model_state"])
                        print(f"Successfully loaded model from {checkpoint_path}")
                        checkpoint_loaded = True
                        break
                    else:
                        # Try to load directly if "model_state" key doesn't exist
                        model.load_state_dict(checkpoint)
                        print(f"Successfully loaded model from {checkpoint_path} (direct loading)")
                        checkpoint_loaded = True
                        break
                except Exception as e:
                    print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            
        if not checkpoint_loaded:
            print(f"No valid checkpoint found in any of the expected locations")
            print("Using uninitialized model weights. Results will not be accurate.")
        
    except Exception as e:
        print(f"Error loading model from DeepLabV3Plus-Pytorch repository: {e}")
        print("Falling back to torchvision's DeepLabV3 model")
        
        # Use torchvision's DeepLabV3 model as a fallback
        import torchvision.models.segmentation as segmentation
        # Print the version of torchvision
        import torchvision
        print(f"Using torchvision version: {torchvision.__version__}")
        
        # Try with COCO weights if available
        try:
            # For newer versions of torchvision
            try:
                from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights
                weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
                model = segmentation.deeplabv3_resnet101(weights=weights)
                print("Using torchvision's DeepLabV3 with ResNet101 backbone and COCO+VOC labels")
            except (ImportError, AttributeError):
                # For older versions
                model = segmentation.deeplabv3_resnet101(pretrained=True)
                print("Using torchvision's DeepLabV3 with ResNet101 backbone (pretrained)")
        except Exception as e1:
            try:
                # Try ResNet50 if ResNet101 fails
                model = segmentation.deeplabv3_resnet50(pretrained=True)
                print("Using torchvision's DeepLabV3 with ResNet50 backbone (pretrained)")
            except Exception as e2:
                print(f"Error loading ResNet101 model: {e1}")
                print(f"Error loading ResNet50 model: {e2}")
                print("Using lightweight MobileNetV3 backbone as last resort")
                model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        
        print("\n⚠️ WARNING: Using torchvision's DeepLabV3 model!")
        print("This model uses different class labels from the Cityscapes dataset.")
        print("The segmentation and ground mask results may be less accurate.")
        print("To improve results, install the DeepLabV3Plus-Pytorch repository.")
    
    model.to(device)
    model.eval()
    return model

def process_image(model, image_path, resize_factor=None, smooth_output=True, min_region_size=100):
    """
    Process a single image and return the segmentation mask
    
    Args:
        model: The segmentation model
        image_path: Path to the input image
        resize_factor: Optional factor to resize the image for coarser segmentation
                      (e.g., 0.5 will make the image half size)
        smooth_output: Whether to apply post-processing to smooth the segmentation
        min_region_size: Minimum size (in pixels) for a region to be preserved
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # Store original size for later upscaling
    
    # Optionally resize the image for coarser segmentation (this is the first way to reduce detail)
    # A smaller resize_factor (e.g., 0.25 or 0.5) will result in a coarser segmentation
    if resize_factor is not None and resize_factor > 0 and resize_factor < 1:
        new_width = int(img.width * resize_factor)
        new_height = int(img.height * resize_factor)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        img_resized = img
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    input_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        
        # Check output format - different model versions return different formats
        if isinstance(output, dict) and 'out' in output:
            output = output['out'][0]  # torchvision DeepLabV3 format
        else:
            output = output[0]  # Original DeepLabV3Plus-Pytorch format
    
    # Process prediction
    output = output.cpu().numpy()
    pred = np.argmax(output, axis=0)
    
    # If we resized the image, resize the prediction back to original size
    if resize_factor is not None and resize_factor > 0 and resize_factor < 1:
        pred_pil = Image.fromarray(pred.astype(np.uint8))
        pred_pil = pred_pil.resize(original_size, Image.Resampling.NEAREST)
        pred = np.array(pred_pil)
    
    # Apply post-processing to smooth the segmentation if requested
    if smooth_output:
        # Create a copy to avoid modifying the original during processing
        smooth_pred = pred.copy()
        
        # Get unique class IDs
        unique_classes = np.unique(pred)
        
        # Process each class
        for class_id in unique_classes:
            # Create binary mask for this class
            class_mask = (pred == class_id).astype(np.uint8)
            
            # 1. Apply morphological operations to remove noise and smooth edges
            # First dilate then erode (closing operation) to close small holes
            class_mask = ndimage.binary_closing(class_mask, structure=np.ones((5, 5))).astype(np.uint8)
            
            # 2. Remove small isolated regions using connected component analysis
            if min_region_size > 0:
                # Label connected components
                labeled_mask, num_features = ndimage.label(class_mask)
                
                # Calculate the size of each component
                component_sizes = ndimage.sum(class_mask, labeled_mask, range(1, num_features + 1))
                
                # Remove components that are smaller than min_region_size
                too_small = component_sizes < min_region_size
                too_small_mask = too_small[labeled_mask - 1]
                labeled_mask[too_small_mask] = 0
                
                # Restore the class mask (now with small regions removed)
                class_mask = (labeled_mask > 0).astype(np.uint8)
                
            # 3. Apply additional smoothing via a Gaussian filter
            class_mask = ndimage.gaussian_filter(class_mask.astype(float), sigma=1.0) > 0.5
            
            # Update the smoothed prediction with this class
            smooth_pred[class_mask] = class_id
        
        return img, smooth_pred
    
    return img, pred

def remove_small_components(mask, min_area_ratio=0.1, relative_size_threshold=0.0):
    """
    Removes small disconnected components from a binary mask to clean up segmentation results
    
    This function performs connected component analysis on a binary mask and removes
    isolated regions that are smaller than a specified percentage of the total image area.
    It also can optionally apply a secondary criterion to preserve regions that might be
    small but still significant relative to the largest component in the mask.
    
    Args:
        mask: Binary mask (0 and 1) where 1 represents the foreground
        min_area_ratio: Minimum area threshold as a ratio of the total image size
                       (default: 0.1, meaning 10% of the total image area)
                       - Smaller values (e.g., 0.001) preserve more details
                       - Larger values (e.g., 0.2) remove more regions
        relative_size_threshold: Secondary threshold as a ratio of the largest component
                       (default: 0.0, meaning no secondary threshold)
                       - Set to 0 to only use the min_area_ratio threshold
                       - Values like 0.2 will preserve regions that are at least 20% 
                         of the largest component's size, even if they're smaller than min_area_ratio
    
    Returns:
        Cleaned binary mask with disconnected components removed
    
    Example:
        # To remove all components less than 10% of image area:
        clean_mask = remove_small_components(mask, min_area_ratio=0.1, relative_size_threshold=0.0)
        
        # To remove components less than 5% of image area, unless they're at least
        # 30% as large as the largest component:
        clean_mask = remove_small_components(mask, min_area_ratio=0.05, relative_size_threshold=0.3)
    """
    # Calculate minimum area in pixels based on the image size
    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = int(total_pixels * min_area_ratio)
    
    # Label connected components in the mask
    labeled_mask, num_features = ndimage.label(mask)
    
    # If there are no features, return the original mask
    if num_features == 0:
        return mask
    
    # Calculate the area of each labeled region
    component_sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
    
    # Create a clean mask containing only components larger than min_area
    clean_mask = np.zeros_like(mask)
    
    # Find the largest component as the "main" region
    largest_component_idx = np.argmax(component_sizes) + 1  # +1 because labels start at 1
    largest_component_size = component_sizes[largest_component_idx - 1]
    
    # Get centroid of the largest component (approximate center)
    largest_component_mask = (labeled_mask == largest_component_idx)
    if np.sum(largest_component_mask) > 0:
        largest_y, largest_x = ndimage.center_of_mass(largest_component_mask)
    else:
        largest_y, largest_x = mask.shape[0]/2, mask.shape[1]/2  # Default to center if no component
    
    # Copy components based on size criteria
    for i, size in enumerate(component_sizes):
        component_idx = i + 1  # Label indexes start at 1
        
        # Determine if we keep this component based on its size
        keep_component = False
        
        # Criterion 1: Component is larger than the minimum area threshold
        if size >= min_area:
            keep_component = True
            
        # Criterion 2: Component is significant relative to the largest component
        # (only if relative_size_threshold > 0)
        elif relative_size_threshold > 0 and size >= (largest_component_size * relative_size_threshold):
            keep_component = True
        
        # If component passed any criterion, add it to the clean mask
        if keep_component:
            clean_mask[labeled_mask == component_idx] = 1
            
    # For very small images or highly fragmented masks, ensure we don't remove everything
    if np.sum(clean_mask) == 0 and num_features > 0:
        # If nothing remains, at least keep the largest component
        clean_mask[labeled_mask == largest_component_idx] = 1
    
    return clean_mask

def extract_ground_mask(segmentation_map, min_area_ratio=0.1, relative_size_threshold=0.0):
    """
    Extract only the ground-related classes from the segmentation map
    
    For Cityscapes model:
    - road (0), sidewalk (1), terrain (9)
    
    For COCO/VOC model (when DeepLabV3Plus-Pytorch is missing):
    - Class 0 may represent background, not road in the COCO labels
    
    Args:
        segmentation_map: The segmentation map with class IDs
        min_area_ratio: Minimum component size as a ratio of image area (default: 0.1 or 10%)
                       Components smaller than this will be removed
        relative_size_threshold: Secondary threshold as ratio of largest component size
                       Set to 0.0 to disable (strict area filtering)
    
    Returns:
        Binary mask where 1 is ground and 0 is not ground
    """
    # Create empty mask
    ground_mask = np.zeros_like(segmentation_map)
    
    # Check if we have a model mismatch by examining the unique class IDs
    unique_classes = np.unique(segmentation_map)
    print(f"DEBUG: Unique class IDs in segmentation map: {unique_classes}")
    
    # If all pixels are labeled as class 0, we likely have a fallback to torchvision model
    # In this case, we need to detect the ground differently
    if len(unique_classes) <= 2 and 0 in unique_classes:
        print("WARNING: All pixels classified as a single class.")
        print("Likely using torchvision's DeepLabV3 model with COCO/VOC labels.")
        print("Attempting to detect ground regions using image characteristics...")
        
        # For COCO/VOC models, try to detect ground by looking at the bottom portion of the image
        h, w = segmentation_map.shape
        bottom_portion = int(h * 0.7)  # Consider bottom 30% as potential ground
        
        # Create a mask that gradually increases the probability of being ground
        # as we move toward the bottom of the image
        y_coords = np.arange(h).reshape(-1, 1)
        gradient_mask = np.zeros_like(segmentation_map, dtype=float)
        
        # Create a gradient where bottom pixels are more likely to be ground
        for y in range(h):
            # Non-linear gradient - emphasize bottom portion
            if y > bottom_portion:
                weight = 0.7 + 0.3 * ((y - bottom_portion) / (h - bottom_portion))
                gradient_mask[y, :] = weight
        
        # Convert to binary mask with reasonable threshold
        ground_mask = (gradient_mask > 0.65).astype(np.uint8)
    else:
        # For Cityscapes model - use the proper class IDs
        # Standard approach: use road (0), sidewalk (1), terrain (9)
        ground_classes = [0, 1, 9]  
        for class_id in ground_classes:
            if class_id in unique_classes:  # Only use classes that actually exist
                ground_mask[segmentation_map == class_id] = 1
    
    # Apply morphological closing to fill small gaps and smooth the mask
    ground_mask = ndimage.binary_closing(ground_mask, structure=np.ones((7, 7))).astype(np.uint8)
    
    # Remove small disconnected components with strict area filtering (no secondary threshold)
    ground_mask = remove_small_components(ground_mask, 
                                          min_area_ratio=min_area_ratio,
                                          relative_size_threshold=relative_size_threshold)
    
    # Safety check - if mask is completely blank, use a basic heuristic
    if np.sum(ground_mask) == 0:
        print("WARNING: No ground detected. Using fallback heuristic...")
        h, w = segmentation_map.shape
        # Create a simple ground mask in the bottom third of the image
        ground_mask[int(h*2/3):, :] = 1
    
    return ground_mask

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
    return np.array(cityscapes_colormap)

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

def visualize_component_removal(mask, min_area_ratio=0.1, relative_size_threshold=0.0):
    """
    Visualize the effect of small component removal on a binary mask
    
    Creates a three-panel visualization showing:
    1. Original mask before component removal
    2. Cleaned mask after component removal
    3. Difference between the two (highlighting what was removed)
    
    Args:
        mask: Binary mask (0 and 1)
        min_area_ratio: Minimum area ratio for component removal
                       (components smaller than this fraction of the image size will be removed)
        relative_size_threshold: Secondary threshold as a ratio of the largest component
                       (components larger than this ratio of the largest component will be kept)
        
    Returns:
        Tuple of (original_mask, cleaned_mask)
    """
    # Get the cleaned mask
    cleaned_mask = remove_small_components(mask, min_area_ratio, relative_size_threshold)
    
    # Create an RGB visualization to highlight differences
    h, w = mask.shape
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Original mask in green
    viz[mask == 1, 1] = 255
    
    # Removed parts in red (parts that were in original but not in cleaned)
    removed = (mask == 1) & (cleaned_mask == 0)
    viz[removed, 0] = 255
    viz[removed, 1] = 0  # Turn off green for removed parts
    
    # Analyze the connected components in the original and cleaned masks
    labeled_orig, num_orig = ndimage.label(mask)
    labeled_clean, num_clean = ndimage.label(cleaned_mask)
    labeled_removed, num_removed = ndimage.label(removed)
    
    # Calculate statistics
    total_pixels = mask.sum()
    removed_pixels = removed.sum()
    kept_pixels = cleaned_mask.sum()
    total_image_pixels = mask.shape[0] * mask.shape[1]
    
    if total_pixels > 0:
        removed_percent = (removed_pixels / total_pixels) * 100
    else:
        removed_percent = 0
    
    # Calculate size statistics of original components
    if num_orig > 0:
        component_sizes = ndimage.sum(mask, labeled_orig, range(1, num_orig + 1))
        largest_size = component_sizes.max()
        largest_percent = (largest_size / total_image_pixels) * 100
        threshold_in_pixels = total_image_pixels * min_area_ratio
    else:
        largest_size = 0
        largest_percent = 0
        threshold_in_pixels = 0
    
    # Display the comparison
    plt.figure(figsize=(15, 6))
    plt.suptitle(f'Component Removal Analysis', fontsize=16)
    
    plt.subplot(1, 3, 1)
    plt.title(f'Original Ground Mask\n{num_orig} components, {total_pixels} pixels')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f'Cleaned Ground Mask\n{num_clean} components, {kept_pixels} pixels')
    plt.imshow(cleaned_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f'Removed Components\nGreen: Kept, Red: Removed')
    plt.imshow(viz)
    plt.axis('off')
    
    # Add detailed statistics at the bottom
    info_text = [
        f"Threshold: Components < {min_area_ratio*100:.1f}% of image area ({threshold_in_pixels:.0f} pixels) are removed",
        f"Removed: {num_removed} components, {removed_pixels} pixels ({removed_percent:.1f}% of original mask)",
        f"Largest component: {largest_size:.0f} pixels ({largest_percent:.1f}% of image area)"
    ]
    
    if relative_size_threshold > 0:
        info_text.append(f"Secondary threshold: Components > {relative_size_threshold*100:.0f}% of largest component are preserved")
    
    plt.figtext(0.5, 0.02, '\n'.join(info_text), ha='center', fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.8, pad=5, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # Make room for the text
    plt.savefig('component_removal_comparison.png', bbox_inches='tight')
    plt.show()
    
    return mask, cleaned_mask

def visualize_results(image, segmentation_map, ground_mask):
    """
    Visualize the original image, segmentation map and ground mask with class labels
    """
    # Get colormap
    colormap = get_cityscapes_colormap()
    
    # Apply colormap to segmentation map
    colored_segmentation = apply_colormap(segmentation_map, colormap)
    
    # Define class labels - check if using Cityscapes or COCO
    unique_classes = np.unique(segmentation_map)
    using_cityscapes = True
    
    if len(unique_classes) <= 3 and np.max(unique_classes) <= 1:
        # Likely using COCO/VOC labels
        using_cityscapes = False
        label_map = {
            0: "background", 
            1: "foreground"
        }
        print("Using COCO/VOC label map for visualization")
    else:
        # Using Cityscapes labels
        label_map = {
            0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole", 6: "traffic light",
            7: "traffic sign", 8: "vegetation", 9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
            14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
        }
        print("Using Cityscapes label map for visualization")
    # Create figure with adjusted size to accommodate the legend
    plt.figure(figsize=(20, 7))
    
    # Original image subplot
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    # Segmentation map subplot with class labels
    plt.subplot(1, 3, 2)
    plt.title('Segmentation Map with Labels')
    plt.imshow(colored_segmentation)
    
    # Find unique classes in the segmentation map
    unique_classes = np.unique(segmentation_map)
    
    # Create a legend for the colors and classes
    legend_patches = []
    for class_id in unique_classes:
        if class_id in label_map:
            # Calculate the center of mass for this class
            binary_map = (segmentation_map == class_id)
            if np.sum(binary_map) > 100:  # Only process regions of substantial size
                # Use scipy.ndimage to find center of mass
                center_y, center_x = ndimage.center_of_mass(binary_map)
                
                # Choose text color that contrasts with the background
                color_rgb = colormap[class_id]
                brightness = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
                text_color = 'black' if brightness > 128 else 'white'
                
                # Add text with a contrasting outline
                plt.annotate(
                    label_map[class_id],
                    (center_x, center_y),
                    color=text_color,
                    fontsize=10,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='none', edgecolor=text_color, pad=1)
                )
            
            # Create patch for legend
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=colormap[class_id]/255.0, 
                                  label=f"{class_id}: {label_map[class_id]}"))
    
    plt.axis('off')
    
    # Ground mask subplot
    plt.subplot(1, 3, 3)
    plt.title('Ground Mask')
    plt.imshow(ground_mask, cmap='gray')
    plt.axis('off')
    
    # Add legend outside the subplots
    plt.figlegend(handles=legend_patches, loc='lower center', ncol=5, 
                 bbox_to_anchor=(0.5, 0), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    plt.savefig('segmentation_results.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=============================================")
    print("DeepLabV3+ Ground Mask Extraction with Enhanced Cleaning")
    print("=============================================")
    print("Loading DeepLabV3+ model with MobileNet backbone...")
    model = get_model()
    
    print(f"\nProcessing image: {IMAGE_PATH}")
    
    # Process the image with smoothing options
    # These parameters control the level of detail and smoothness:
    resize_factor = 0.6        # Lower = coarser segmentation (range: 0.3-1.0)
    min_region_size = 400      # Higher = fewer small regions (range: 100-1000)
    
    # Component removal parameters
    min_area_ratio = 0.1       # Components smaller than this % of image area will be removed (0.1 = 10%)
    relative_size_threshold = 0.0  # Secondary size threshold (0.0 = disabled, strict filtering by area only)
    
    print(f"\nApplying segmentation with smoothing parameters:")
    print(f"- resize_factor = {resize_factor} (lower values = coarser segmentation)")
    print(f"- min_region_size = {min_region_size} pixels (higher values = fewer isolated regions)")
    print(f"- component removal threshold = {min_area_ratio*100:.1f}% of image area (strict filtering)")
    
    # Step 1: Run the segmentation model
    img, segmentation_map = process_image(model, IMAGE_PATH, 
                                          resize_factor=resize_factor,
                                          smooth_output=True,
                                          min_region_size=min_region_size)
        
    # Step 2: Create the initial ground mask without small component removal
    print("\nExtracting ground-related classes (road, sidewalk, terrain)...")
    ground_mask_raw = np.zeros_like(segmentation_map)
    ground_classes = [0, 1, 9]  # road, sidewalk, terrain
    for class_id in ground_classes:
        ground_mask_raw[segmentation_map == class_id] = 1
        
    # Apply morphological closing for smoothing
    print("Applying morphological operations for initial smoothing...")
    kernel_size = 7  # Can be increased for more aggressive smoothing
    ground_mask_smoothed = ndimage.binary_closing(ground_mask_raw, 
                                                structure=np.ones((kernel_size, kernel_size))).astype(np.uint8)
    
    # Step 3: Generate masks with different cleaning parameters for comparison
    print("Generating masks with different cleaning parameters...")
    
    # Original smoothed mask without component removal
    ground_mask_no_removal = ground_mask_smoothed.copy()
    
    # Standard cleaning (10% threshold, strict filtering)
    ground_mask_standard = extract_ground_mask(segmentation_map, 
                                              min_area_ratio=0.1, 
                                              relative_size_threshold=0.0)
    
    # Aggressive cleaning (15% threshold, strict filtering)
    ground_mask_aggressive = extract_ground_mask(segmentation_map, 
                                                min_area_ratio=0.15, 
                                                relative_size_threshold=0.0)
    
    # Step 4: Visualize the effect of different cleaning parameters
    print("\nVisualizing the effect of component removal with different parameters...")
    print("1. Standard cleaning (components < 10% of image area)...")
    visualize_component_removal(ground_mask_smoothed, min_area_ratio=0.1, relative_size_threshold=0.0)
    
    print("\n2. Aggressive cleaning (components < 15% of image area)...")
    visualize_component_removal(ground_mask_smoothed, min_area_ratio=0.15, relative_size_threshold=0.0)
    
    # Step 5: Save the final selected ground mask (using the aggressive version)
    mask_filename = 'ground_mask.png'
    Image.fromarray((ground_mask_aggressive * 255).astype(np.uint8)).save(mask_filename)
    
    # Step 6: Visualize segmentation results with the cleaned ground mask
    print("Generating visualization of segmentation results...")
    visualize_results(img, segmentation_map, ground_mask_aggressive)
    
    # Provide information about the results
    print("\n=============================================")
    print(f"Processing complete! Results saved as:")
    print(f"- 'segmentation_results.png': Original image, class segmentation, and final ground mask")
    print(f"- 'component_removal_comparison.png': Before/after component removal")
    print(f"- '{mask_filename}': Final binary ground mask (white = ground)")
    print("=============================================")
    print("\nTo customize the component removal:")
    print("1. Increase min_area_ratio (e.g., 0.15 or 0.2) to remove more/larger components")
    print("2. Decrease min_area_ratio (e.g., 0.05 or 0.01) to preserve more components")
    print("3. Set relative_size_threshold > 0 (e.g., 0.3) to preserve components based on")
    print("   their relative size compared to the largest component")
    print("4. For smoother masks, increase kernel_size for morphological operations")
    print("   or decrease resize_factor for initial segmentation")
    print("=============================================")
