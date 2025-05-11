
import torch
import torch.nn as nn
import torchvision.models as models

def deeplabv3plus_mobilenet(num_classes=19, output_stride=16, pretrained_backbone=True):
    """
    Create a temporary fallback that uses torchvision's models but with the correct API
    """
    print("Creating simplified DeepLabV3+ model using torchvision as a fallback")
    # Use torchvision's DeepLabV3 model directly
    try:
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        
        # Adjust the classifier to have the correct number of output classes
        in_channels = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        # Mention that this is a simplified version
        print("WARNING: Using a simplified model. For best results, install the full DeepLabV3Plus-Pytorch repo.")
        
        return model
    except:
        # If that fails, try mobilenet backbone
        try:
            model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
            # Adjust the classifier to have the correct number of output classes
            in_channels = model.classifier[-1].in_channels
            model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            return model
        except:
            print("Failed to create DeepLabV3 model using torchvision")
            raise
