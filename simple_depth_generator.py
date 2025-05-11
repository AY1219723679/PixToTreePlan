import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def generate_depth_map():
    print("Starting depth map generation")
    
    # Ensure the output directory exists
    os.makedirs("output_depth", exist_ok=True)
    
    # Path to the cutout image
    cutout_path = "cutout_ground.png"
    print(f"Processing image: {cutout_path}")
    
    try:
        # Load the cutout image (with alpha channel)
        cutout = Image.open(cutout_path)
        print(f"Image loaded successfully. Size: {cutout.size}, Mode: {cutout.mode}")
        
        # Create a black background image
        rgb_image = Image.new("RGB", cutout.size, (0, 0, 0))
        
        # Only paste where the alpha channel is non-zero
        if cutout.mode == 'RGBA':
            rgb_image.paste(cutout, mask=cutout.split()[3])
            print("Applied alpha mask to create RGB image")
        else:
            rgb_image.paste(cutout)
            print("Image doesn't have alpha channel, using as is")
        
        # Save the RGB version for processing
        temp_path = "temp_for_depth.jpg"
        rgb_image.save(temp_path)
        print(f"Saved temporary RGB image to {temp_path}")
        
        # Load MiDaS model
        print("Loading MiDaS model...")
        try:
            # Use MiDaS v2 model from PyTorch Hub
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            print(f"Using device: {device}")
            
            midas.to(device)
            midas.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.small_transform
            
            # Load image
            img = cv2.imread(temp_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply input transforms
            input_batch = transform(img).to(device)
            
            # Run the model
            with torch.no_grad():
                prediction = midas(input_batch)
                
                # Interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            output = prediction.cpu().numpy()
            
            # Normalize output
            output_normalized = (output - output.min()) / (output.max() - output.min())
            
            # Apply the alpha mask to the depth map (for masked version)
            if cutout.mode == 'RGBA':
                alpha = np.array(cutout.split()[3])
                mask = alpha > 0
                
                # Create masked depth map
                output_masked = output_normalized.copy()
                output_masked[~mask] = 0  # Set depth to 0 where alpha is 0
                
                # Save the masked depth map
                plt.imsave("output_depth/depth_masked.png", output_masked, cmap="inferno")
                print("Saved masked depth map to output_depth/depth_masked.png")
            
            # Save the full depth map
            plt.imsave("output_depth/depth_map.png", output_normalized, cmap="inferno")
            print("Saved depth map to output_depth/depth_map.png")
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            plt.subplot(1, 3, 1)
            plt.title("Original Cutout")
            plt.imshow(cutout)
            plt.axis("off")
            
            plt.subplot(1, 3, 2)
            plt.title("Depth Map (brighter = closer)")
            plt.imshow(output_normalized, cmap="inferno")
            plt.axis("off")
            
            if cutout.mode == 'RGBA':
                plt.subplot(1, 3, 3)
                plt.title("Masked Depth Map")
                plt.imshow(output_masked, cmap="inferno")
                plt.axis("off")
            
            plt.tight_layout()
            plt.savefig("output_depth/depth_visualization.png")
            print("Saved visualization to output_depth/depth_visualization.png")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            print("Depth map generation complete!")
            
        except Exception as e:
            print(f"Error in MiDaS processing: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== MiDaS Depth Map Generator ===")
    generate_depth_map()
