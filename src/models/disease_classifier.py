"""Disease classifier wrapper for inference and prediction."""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

from .cnn_model import PlantDiseaseCNN, load_model


class DiseaseClassifier:
    """Wrapper class for plant disease classification."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model: Optional[PlantDiseaseCNN] = None,
        class_names: Optional[List[str]] = None,
        device: str = "cpu"
    ) -> None:
        """
        Initialize the disease classifier.
        
        Args:
            model_path: Path to saved model (if loading from file)
            model: Pre-trained model instance
            class_names: List of class names
            device: Device to run inference on
        """
        self.device = device
        self.class_names = class_names or []
        
        if model_path and Path(model_path).exists():
            self.model = self._load_model_from_path(model_path)
        elif model:
            self.model = model.to(device)
        else:
            raise ValueError("Either model_path or model must be provided")
        
        self.model.eval()
        
        # Define image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model_from_path(self, model_path: Path) -> PlantDiseaseCNN:
        """Load model from saved checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model parameters
        num_classes = checkpoint.get('num_classes', len(self.class_names))
        model_name = checkpoint.get('model_name', 'resnet18')
        
        # Create model
        model = PlantDiseaseCNN(
            num_classes=num_classes,
            model_name=model_name
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update class names if available
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        
        return model.to(self.device)
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5
    ) -> List[Dict[str, float]]:
        """
        Predict disease from image.
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with class names and confidence scores
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            
            # Convert to list of dictionaries
            predictions = []
            for i in range(top_k):
                idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                
                class_name = (
                    self.class_names[idx] 
                    if idx < len(self.class_names) 
                    else f"Class_{idx}"
                )
                
                predictions.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'class_index': idx
                })
        
        return predictions
    
    def predict_single(
        self,
        image: Image.Image
    ) -> Tuple[str, float]:
        """
        Get single top prediction.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (class_name, confidence)
        """
        predictions = self.predict(image, top_k=1)
        return predictions[0]['class_name'], predictions[0]['confidence']
    
    def predict_from_path(
        self,
        image_path: Path,
        top_k: int = 5
    ) -> List[Dict[str, float]]:
        """
        Predict disease from image file path.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with class names and confidence scores
        """
        image = Image.open(image_path)
        return self.predict(image, top_k)
    
    def get_disease_info(self, class_name: str) -> Dict[str, str]:
        """
        Get information about a specific disease.
        
        Args:
            class_name: Name of the disease class
            
        Returns:
            Dictionary with disease information
        """
        # This would typically come from a database or config file
        disease_info = {
            "rice_blast": {
                "name": "Rice Blast",
                "symptoms": "Small, diamond-shaped lesions on leaves, stems, and panicles",
                "treatment": "Use resistant varieties, proper water management, fungicide application",
                "prevention": "Avoid excessive nitrogen, maintain proper spacing, crop rotation"
            },
            "wheat_rust": {
                "name": "Wheat Rust",
                "symptoms": "Orange or yellow pustules on leaves and stems",
                "treatment": "Fungicide application, resistant varieties",
                "prevention": "Crop rotation, proper field sanitation, early planting"
            },
            "maize_leaf_blight": {
                "name": "Maize Leaf Blight",
                "symptoms": "Large, tan lesions on leaves with dark borders",
                "treatment": "Fungicide application, resistant varieties",
                "prevention": "Crop rotation, proper irrigation, field sanitation"
            },
            "healthy": {
                "name": "Healthy Plant",
                "symptoms": "No visible disease symptoms",
                "treatment": "Continue current care practices",
                "prevention": "Maintain good agricultural practices"
            }
        }
        
        return disease_info.get(class_name.lower(), {
            "name": class_name,
            "symptoms": "Information not available",
            "treatment": "Consult local agricultural extension",
            "prevention": "Follow good agricultural practices"
        })
