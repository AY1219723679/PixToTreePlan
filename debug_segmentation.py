import sys
import os
import torch

# Add current directory to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)
sys.path.append(os.path.join(curr_dir, "main", "get_ground_mask"))

# Print debug info
print(f"Python version: {sys.version}")
print(f"Current directory: {curr_dir}")
print(f"PyTorch version: {torch.__version__}")

# Check for DeepLabV3Plus-Pytorch repository
deeplab_path = os.path.join(curr_dir, "DeepLabV3Plus-Pytorch")
print(f"DeepLabV3Plus-Pytorch path: {deeplab_path}")
print(f"Directory exists: {os.path.isdir(deeplab_path)}")
print(f"Directory contents: {os.listdir(deeplab_path) if os.path.isdir(deeplab_path) else 'DIRECTORY NOT FOUND'}")

# Check checkpoint path
checkpoint_path = os.path.join(curr_dir, "checkpoints", "best_deeplabv3plus_mobilenet_cityscapes_os16.pth")
print(f"Checkpoint path: {checkpoint_path}")
print(f"File exists: {os.path.isfile(checkpoint_path)}")

# Try importing segmentation models
try:
    import torchvision.models.segmentation as segmentation
    print("Successfully imported torchvision.models.segmentation")
    
    # Print available models
    print(f"Available segmentation models: {dir(segmentation)}")
    
    # Try creating the fallback model
    model = segmentation.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
    print("Successfully created DeepLabV3 model with COCO/VOC weights")
    print(f"Model class names: {model.aux_classifier[4].bias.shape[0] if hasattr(model, 'aux_classifier') else 'Unknown'}")
    
except Exception as e:
    print(f"Error: {e}")

print("\nDebug complete.")
