"""Mock data for plant disease detection app."""

import random
from typing import List, Dict, Any
from PIL import Image
import numpy as np


# Mock disease categories and information
DISEASE_CATEGORIES = {
    "rice": {
        "healthy": {
            "name": "Healthy Rice",
            "symptoms": "Green, healthy leaves with no visible spots or discoloration",
            "treatment": "Continue current care practices",
            "prevention": "Maintain proper irrigation, use disease-resistant varieties, and practice crop rotation"
        },
        "bacterial_leaf_blight": {
            "name": "Bacterial Leaf Blight",
            "symptoms": "Water-soaked lesions on leaves, yellowing, and wilting",
            "treatment": "Apply copper-based fungicides, remove infected plants",
            "prevention": "Use disease-free seeds, avoid overhead irrigation, practice field sanitation"
        },
        "brown_spot": {
            "name": "Brown Spot",
            "symptoms": "Small brown spots on leaves that may coalesce",
            "treatment": "Apply fungicides containing propiconazole or tebuconazole",
            "prevention": "Use resistant varieties, proper spacing, avoid excessive nitrogen"
        },
        "blast": {
            "name": "Rice Blast",
            "symptoms": "Diamond-shaped lesions with gray centers and brown borders",
            "treatment": "Apply tricyclazole or azoxystrobin fungicides",
            "prevention": "Use resistant varieties, avoid excessive nitrogen, proper water management"
        }
    },
    "wheat": {
        "healthy": {
            "name": "Healthy Wheat",
            "symptoms": "Green, erect leaves with no visible disease symptoms",
            "treatment": "Continue current care practices",
            "prevention": "Use certified seeds, proper crop rotation, balanced fertilization"
        },
        "rust": {
            "name": "Wheat Rust",
            "symptoms": "Orange or yellow pustules on leaves and stems",
            "treatment": "Apply fungicides like propiconazole or tebuconazole",
            "prevention": "Use resistant varieties, avoid late planting, proper field sanitation"
        },
        "powdery_mildew": {
            "name": "Powdery Mildew",
            "symptoms": "White powdery coating on leaves and stems",
            "treatment": "Apply sulfur-based fungicides or systemic fungicides",
            "prevention": "Improve air circulation, avoid dense planting, use resistant varieties"
        }
    },
    "tomato": {
        "healthy": {
            "name": "Healthy Tomato",
            "symptoms": "Green, healthy leaves and stems with no visible disease",
            "treatment": "Continue current care practices",
            "prevention": "Proper spacing, good air circulation, regular monitoring"
        },
        "early_blight": {
            "name": "Early Blight",
            "symptoms": "Dark brown spots with concentric rings on lower leaves",
            "treatment": "Apply copper-based fungicides or chlorothalonil",
            "prevention": "Crop rotation, proper spacing, avoid overhead watering"
        },
        "late_blight": {
            "name": "Late Blight",
            "symptoms": "Water-soaked lesions that turn brown and papery",
            "treatment": "Apply fungicides containing metalaxyl or cymoxanil",
            "prevention": "Avoid overhead watering, proper spacing, use resistant varieties"
        }
    },
    "potato": {
        "healthy": {
            "name": "Healthy Potato",
            "symptoms": "Green, healthy foliage with no visible disease symptoms",
            "treatment": "Continue current care practices",
            "prevention": "Use certified seed potatoes, proper crop rotation"
        },
        "late_blight": {
            "name": "Potato Late Blight",
            "symptoms": "Dark, water-soaked lesions on leaves and stems",
            "treatment": "Apply fungicides containing metalaxyl or cymoxanil",
            "prevention": "Use resistant varieties, avoid overhead watering, proper spacing"
        },
        "scab": {
            "name": "Potato Scab",
            "symptoms": "Rough, scabby lesions on potato tubers",
            "treatment": "No effective treatment once infected",
            "prevention": "Maintain soil pH 5.2-5.5, use resistant varieties, crop rotation"
        }
    }
}

# Mock prediction results
def generate_mock_predictions(crop_type: str = None, num_predictions: int = 5) -> List[Dict[str, Any]]:
    """Generate mock prediction results."""
    if crop_type is None:
        crop_type = random.choice(list(DISEASE_CATEGORIES.keys()))
    
    diseases = list(DISEASE_CATEGORIES[crop_type].keys())
    
    # If we need more predictions than diseases available, repeat some diseases
    if num_predictions > len(diseases):
        # Repeat diseases to get enough predictions
        extended_diseases = diseases * ((num_predictions // len(diseases)) + 1)
        diseases = extended_diseases[:num_predictions]
    
    # Generate random confidence scores that sum to approximately 1.0
    confidences = np.random.dirichlet(np.ones(len(diseases)))
    confidences = confidences * 0.8 + 0.1  # Scale to 0.1-0.9 range
    confidences = np.sort(confidences)[::-1]  # Sort in descending order
    
    predictions = []
    for i, disease in enumerate(diseases[:num_predictions]):
        predictions.append({
            'class_name': f"{crop_type}_{disease}",
            'confidence': float(confidences[i]),
            'crop_type': crop_type,
            'disease': disease
        })
    
    return predictions


def get_mock_disease_info(class_name: str) -> Dict[str, str]:
    """Get mock disease information for a given class name."""
    try:
        crop_type, disease = class_name.split('_', 1)
        if crop_type in DISEASE_CATEGORIES and disease in DISEASE_CATEGORIES[crop_type]:
            return DISEASE_CATEGORIES[crop_type][disease]
    except ValueError:
        pass
    
    # Default fallback
    return {
        "name": "Unknown Disease",
        "symptoms": "Symptoms information not available",
        "treatment": "Please consult with a local agricultural expert",
        "prevention": "General prevention practices: crop rotation, proper spacing, regular monitoring"
    }


def create_mock_image_info(image: Image.Image) -> Dict[str, Any]:
    """Create mock image information."""
    return {
        'width': image.width,
        'height': image.height,
        'format': image.format,
        'mode': image.mode,
        'size_mb': round(len(image.tobytes()) / (1024 * 1024), 2),
        'aspect_ratio': round(image.width / image.height, 2)
    }


def get_mock_model_performance() -> Dict[str, float]:
    """Get mock model performance metrics."""
    return {
        'accuracy': random.uniform(0.85, 0.95),
        'precision': random.uniform(0.80, 0.92),
        'recall': random.uniform(0.82, 0.90),
        'f1_score': random.uniform(0.81, 0.91)
    }


def get_mock_training_history() -> Dict[str, List[float]]:
    """Get mock training history data."""
    epochs = 50
    return {
        'epochs': list(range(1, epochs + 1)),
        'train_loss': [random.uniform(0.1, 2.0) * np.exp(-i/20) for i in range(epochs)],
        'val_loss': [random.uniform(0.15, 2.2) * np.exp(-i/18) for i in range(epochs)],
        'train_acc': [min(0.99, 0.5 + i * 0.01) for i in range(epochs)],
        'val_acc': [min(0.95, 0.45 + i * 0.009) for i in range(epochs)]
    }


def get_supported_crops() -> List[str]:
    """Get list of supported crop types."""
    return list(DISEASE_CATEGORIES.keys())


def get_crop_diseases(crop_type: str) -> List[str]:
    """Get list of diseases for a specific crop."""
    if crop_type in DISEASE_CATEGORIES:
        return list(DISEASE_CATEGORIES[crop_type].keys())
    return []


def create_mock_sample_images() -> Dict[str, List[str]]:
    """Create mock sample image paths."""
    sample_images = {}
    for crop_type in DISEASE_CATEGORIES.keys():
        diseases = list(DISEASE_CATEGORIES[crop_type].keys())
        sample_images[crop_type] = [
            f"sample_{disease}_{i}.jpg" 
            for disease in diseases 
            for i in range(1, 4)  # 3 samples per disease
        ]
    return sample_images


# Mock model class for demonstration
class MockDiseaseClassifier:
    """Mock disease classifier for demonstration purposes."""
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.is_loaded = True
    
    def predict(self, image: Image.Image, top_k: int = 5) -> List[Dict[str, Any]]:
        """Mock prediction method."""
        # Simulate processing time
        import time
        time.sleep(0.5)
        
        # Generate mock predictions
        crop_type = random.choice(get_supported_crops())
        predictions = generate_mock_predictions(crop_type, top_k)
        
        return predictions
    
    def get_disease_info(self, class_name: str) -> Dict[str, str]:
        """Get disease information."""
        return get_mock_disease_info(class_name)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'CNN (ResNet-50)',
            'input_size': (224, 224),
            'num_classes': 15,
            'training_accuracy': random.uniform(0.85, 0.95),
            'validation_accuracy': random.uniform(0.80, 0.90),
            'model_size_mb': random.uniform(50, 100)
        }
