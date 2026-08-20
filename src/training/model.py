import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Instantiates and returns a PyTorch model based on the given architecture name.
    """
    if architecture.lower() == "resnet18":
        # Load pre-trained ResNet-18 or default weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Replace the final fully connected layer to match CIFAR-10 class count
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")