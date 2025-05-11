import torch
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def generate_depth_from_cutout(cutout_path, output_path=None, save_visualization=True):
    """
    Generate a depth map from a cutout image using MiDaS v2 model.
    
    Args:
        cutout_path (str): Path to the cutout image with transparency
        output_path (str, optional): Path to save the depth map. If None, will use cutout name with _depth suffix
        save_visualization (bool): Whether to save a colorized visualization of the depth map
    
    Returns:
        numpy.ndarray: The depth map
    """
    print(f"Processing: {cutout_path}")
    
    # Set default output path if not provided
    if output_path is None:
        basename = os.path.splitext(cutout_path)[0]
        output_path = f"{basename}_depth.png"
    
    # Load cutout image with transparency (RGBA)
    cutout = Image.open(cutout_path)
    
    # Convert RGBA to RGB (the model expects RGB)
    # For transparent regions, we'll use black as background
    bg_color = (0, 0, 0)  # Black background
    rgb_image = Image.new("RGB", cutout.size, bg_color)
    rgb_image.paste(cutout, mask=cutout.split()[3])  # Use alpha as mask for pasting
    
    # Save the RGB version of the image as a temporary file
    temp_rgb_path = f"{os.path.splitext(cutout_path)[0]}_temp_rgb.jpg"
    rgb_image.save(temp_rgb_path)
    
    # Create the model with MiDaS v2
    print("Loading MiDaS v2 model...")
    model_type = "DPT_Large"  # MiDaS v3 - best quality (can also try "MiDaS_small" for smaller model)
    
    # Try to load the model (with error handling)
    try:
        # Using PyTorch Hub to load MiDaS
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        
        # Move model to GPU if available
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        
        # Load transforms for preprocessing the image
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
            
        # Load image in MiDaS format using OpenCV
        img = cv2.imread(temp_rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply input transforms
        input_batch = transform(img).to(device)
        
        # Inference
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Convert the depth map to the range 0-65535 (16-bit) for better precision
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_normalized = 65535 * (depth_map - depth_min) / (depth_max - depth_min)
        depth_normalized = depth_normalized.astype(np.uint16)
        
        # Save the depth map as a 16-bit PNG
        cv2.imwrite(output_path, depth_normalized)
        print(f"Depth map saved to: {output_path}")
        
        # Create cutout version of the depth map (only depth where the original has content)
        # Get the alpha channel from the original cutout
        alpha = np.array(cutout.split()[3])
        
        # Apply the alpha mask to the depth map
        masked_depth = depth_map.copy()
        masked_depth[alpha == 0] = 0  # Set transparent parts to 0 depth
        
        # Save the masked depth map
        masked_output_path = f"{os.path.splitext(output_path)[0]}_masked.png"
        masked_depth_normalized = 65535 * (masked_depth - depth_min) / (depth_max - depth_min)
        masked_depth_normalized = masked_depth_normalized.astype(np.uint16)
        masked_depth_normalized[alpha == 0] = 0
        cv2.imwrite(masked_output_path, masked_depth_normalized)
        print(f"Masked depth map saved to: {masked_output_path}")
        
        # Create and save visualization
        if save_visualization:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.title("Original Cutout")
            plt.imshow(cutout)
            plt.axis("off")
            
            plt.subplot(2, 2, 2)
            plt.title("RGB Version")
            plt.imshow(rgb_image)
            plt.axis("off")
            
            plt.subplot(2, 2, 3)
            plt.title("Depth Map (brighter = closer)")
            plt.imshow(depth_map, cmap="plasma")
            plt.axis("off")
            
            plt.subplot(2, 2, 4)
            plt.title("Masked Depth Map")
            plt.imshow(masked_depth, cmap="plasma")
            plt.axis("off")
            
            vis_path = f"{os.path.splitext(output_path)[0]}_visualization.png"
            plt.tight_layout()
            plt.savefig(vis_path)
            print(f"Visualization saved to: {vis_path}")
            
        # Clean up the temporary file
        os.remove(temp_rgb_path)
        
        return masked_depth
        
    except Exception as e:
        print(f"Error processing depth image: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Process the cutout image we created earlier
    cutout_path = "cutout_ground.png"
    
    # Ensure the cutout exists
    if not os.path.exists(cutout_path):
        print(f"Error: {cutout_path} not found")
    else:
        # Create an output directory for depth maps if it doesn't exist
        output_dir = "output_depth"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save the depth map
        output_path = os.path.join(output_dir, "ground_depth.png")
        depth_map = generate_depth_from_cutout(cutout_path, output_path)
        
        print("\nDepth generation complete!")
        print(f"- Depth map saved to: {output_path}")
        print(f"- Masked depth map saved to: {os.path.splitext(output_path)[0]}_masked.png")
        print(f"- Visualization saved to: {os.path.splitext(output_path)[0]}_visualization.png")
