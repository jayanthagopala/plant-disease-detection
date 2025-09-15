"""Tests for model components."""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil

from src.models.cnn_model import PlantDiseaseCNN, create_model, save_model, load_model
from src.models.disease_classifier import DiseaseClassifier


class TestPlantDiseaseCNN:
    """Test cases for PlantDiseaseCNN model."""
    
    def test_model_creation(self):
        """Test model creation with different architectures."""
        for model_name in ["resnet18", "resnet50", "efficientnet_b0"]:
            model = create_model(
                num_classes=10,
                model_name=model_name,
                device="cpu"
            )
            assert isinstance(model, PlantDiseaseCNN)
            assert model.num_classes == 10
            assert model.model_name == model_name
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = create_model(num_classes=5, device="cpu")
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        output = model(input_tensor)
        assert output.shape == (batch_size, 5)
    
    def test_feature_extraction(self):
        """Test feature extraction method."""
        model = create_model(num_classes=5, device="cpu")
        input_tensor = torch.randn(1, 3, 224, 224)
        
        features = model.extract_features(input_tensor)
        assert features.dim() == 2  # Should be 2D (batch_size, feature_dim)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        model = create_model(num_classes=3, device="cpu")
        class_names = ["class1", "class2", "class3"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.pth"
            
            # Save model
            save_model(
                model=model,
                save_path=save_path,
                epoch=10,
                loss=0.5,
                accuracy=85.0,
                class_names=class_names
            )
            
            assert save_path.exists()
            
            # Load model
            loaded_model = load_model(
                model_path=save_path,
                num_classes=3,
                device="cpu"
            )
            
            assert isinstance(loaded_model, PlantDiseaseCNN)
            assert loaded_model.num_classes == 3


class TestDiseaseClassifier:
    """Test cases for DiseaseClassifier."""
    
    def test_classifier_creation_with_model(self):
        """Test classifier creation with a model."""
        model = create_model(num_classes=5, device="cpu")
        classifier = DiseaseClassifier(
            model=model,
            class_names=["class1", "class2", "class3", "class4", "class5"],
            device="cpu"
        )
        
        assert classifier is not None
        assert len(classifier.class_names) == 5
    
    def test_image_preprocessing(self):
        """Test image preprocessing."""
        from PIL import Image
        import numpy as np
        
        model = create_model(num_classes=3, device="cpu")
        classifier = DiseaseClassifier(
            model=model,
            class_names=["class1", "class2", "class3"],
            device="cpu"
        )
        
        # Create a test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test preprocessing
        processed = classifier.preprocess_image(test_image)
        assert processed.shape == (1, 3, 224, 224)
        assert processed.dtype == torch.float32
    
    def test_prediction(self):
        """Test prediction functionality."""
        model = create_model(num_classes=3, device="cpu")
        classifier = DiseaseClassifier(
            model=model,
            class_names=["class1", "class2", "class3"],
            device="cpu"
        )
        
        # Create a test image
        from PIL import Image
        import numpy as np
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test prediction
        predictions = classifier.predict(test_image, top_k=3)
        
        assert len(predictions) == 3
        assert all("class_name" in pred for pred in predictions)
        assert all("confidence" in pred for pred in predictions)
        assert all("class_index" in pred for pred in predictions)
        
        # Test single prediction
        class_name, confidence = classifier.predict_single(test_image)
        assert isinstance(class_name, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__])
