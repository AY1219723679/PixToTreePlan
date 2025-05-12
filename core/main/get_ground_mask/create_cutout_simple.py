import numpy as np
from PIL import Image

def create_cutout_with_mask(image_path, mask_path, output_path):
    """
    Creates a transparent cutout of an image based on a binary mask.
    """
    # Load the original image and the mask
    original = Image.open(image_path)
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
    
    # Ensure the mask and image have the same dimensions
    if original.size != mask.size:
        mask = mask.resize(original.size, Image.NEAREST)
    
    # Convert the original image to RGBA format
    rgba_image = original.convert("RGBA")
    
    # Get the image data as numpy arrays
    rgba_data = np.array(rgba_image)
    mask_data = np.array(mask)
    
    # Normalize mask values to 0 or 1
    if mask_data.max() > 1:
        mask_data = mask_data / 255
    
    # Apply the mask to the alpha channel
    rgba_data[:, :, 3] = (mask_data * 255).astype(np.uint8)
    
    # Create a new image from the modified data
    result = Image.fromarray(rgba_data)
    
    # Save the resulting image
    result.save(output_path, format="PNG")
    print(f"Cutout image saved to {output_path}")

# Only execute demo code when script is run directly, not when imported
if __name__ == "__main__":
    # Use the default paths from your project
    image_path = "input_images/urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg"
    mask_path = "ground_mask.png"
    output_path = "cutout_ground.png"
    
    # Create the cutout
    create_cutout_with_mask(image_path, mask_path, output_path)