"""CNN model for plant disease classification."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
from pathlib import Path


class PlantDiseaseCNN(nn.Module):
    """CNN model for plant disease classification using transfer learning."""
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ) -> None:
        """
        Initialize the plant disease CNN model.
        
        Args:
            num_classes: Number of disease classes to classify
            model_name: Base model architecture ('resnet18', 'resnet50', 'efficientnet_b0')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(PlantDiseaseCNN, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        # Load base model
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone without classification."""
        return self.backbone(x)


def create_model(
    num_classes: int,
    model_name: str = "resnet18",
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    device: str = "cpu"
) -> PlantDiseaseCNN:
    """
    Create and initialize a plant disease CNN model.
    
    Args:
        num_classes: Number of disease classes
        model_name: Base model architecture
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for regularization
        device: Device to load the model on
        
    Returns:
        Initialized PlantDiseaseCNN model
    """
    model = PlantDiseaseCNN(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    return model.to(device)


def load_model(
    model_path: Path,
    num_classes: int,
    model_name: str = "resnet18",
    device: str = "cpu"
) -> PlantDiseaseCNN:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model
        num_classes: Number of disease classes
        model_name: Base model architecture
        device: Device to load the model on
        
    Returns:
        Loaded PlantDiseaseCNN model
    """
    model = create_model(
        num_classes=num_classes,
        model_name=model_name,
        device=device
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def save_model(
    model: PlantDiseaseCNN,
    save_path: Path,
    epoch: int,
    loss: float,
    accuracy: float,
    class_names: list,
    **kwargs
) -> None:
    """
    Save a trained model to file.
    
    Args:
        model: The model to save
        save_path: Path to save the model
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy
        class_names: List of class names
        **kwargs: Additional metadata to save
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'class_names': class_names,
        'model_name': model.model_name,
        'num_classes': model.num_classes,
        **kwargs
    }
    
    torch.save(checkpoint, save_path)
