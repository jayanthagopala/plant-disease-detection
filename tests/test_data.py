"""Tests for data utilities."""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import numpy as np

from src.data.dataset import PlantDiseaseDataset, create_data_loaders
from src.data.preprocessing import ImagePreprocessor, get_transforms
from src.data.utils import create_sample_dataset, analyze_dataset


class TestPlantDiseaseDataset:
    """Test cases for PlantDiseaseDataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation with sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            class_names = ["class1", "class2"]
            
            # Create sample data
            for class_name in class_names:
                class_dir = data_dir / class_name
                class_dir.mkdir(parents=True)
                
                # Create sample images
                for i in range(5):
                    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                    img.save(class_dir / f"sample_{i}.jpg")
            
            # Create dataset
            dataset = PlantDiseaseDataset(
                data_dir=data_dir,
                class_names=class_names
            )
            
            assert len(dataset) == 10  # 5 images per class
            assert len(dataset.class_names) == 2
            assert len(dataset.class_to_idx) == 2
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            class_names = ["class1"]
            
            # Create sample data
            class_dir = data_dir / class_names[0]
            class_dir.mkdir(parents=True)
            
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(class_dir / "sample.jpg")
            
            # Create dataset
            dataset = PlantDiseaseDataset(
                data_dir=data_dir,
                class_names=class_names
            )
            
            # Test getitem
            image, label = dataset[0]
            assert isinstance(image, Image.Image)
            assert isinstance(label, int)
            assert label == 0  # First class
    
    def test_class_distribution(self):
        """Test class distribution calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            class_names = ["class1", "class2"]
            
            # Create sample data with different counts
            for i, class_name in enumerate(class_names):
                class_dir = data_dir / class_name
                class_dir.mkdir(parents=True)
                
                # Create different number of images for each class
                for j in range(3 + i):  # class1: 3 images, class2: 4 images
                    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                    img.save(class_dir / f"sample_{j}.jpg")
            
            # Create dataset
            dataset = PlantDiseaseDataset(
                data_dir=data_dir,
                class_names=class_names
            )
            
            # Test distribution
            distribution = dataset.get_class_distribution()
            assert distribution["class1"] == 3
            assert distribution["class2"] == 4


class TestDataLoaders:
    """Test cases for data loaders."""
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            class_names = ["class1", "class2"]
            
            # Create sample data
            for class_name in class_names:
                class_dir = data_dir / class_name
                class_dir.mkdir(parents=True)
                
                for i in range(10):
                    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                    img.save(class_dir / f"sample_{i}.jpg")
            
            # Create data loaders
            train_loader, val_loader, test_loader, returned_class_names = create_data_loaders(
                data_dir=data_dir,
                class_names=class_names,
                batch_size=4
            )
            
            assert len(train_loader) > 0
            assert len(val_loader) > 0
            assert len(test_loader) > 0
            assert returned_class_names == class_names
            
            # Test batch iteration
            for batch_images, batch_labels in train_loader:
                assert batch_images.shape[0] <= 4  # batch_size
                assert batch_images.shape[1] == 3  # RGB channels
                assert batch_images.shape[2] == 224  # height
                assert batch_images.shape[3] == 224  # width
                assert len(batch_labels) == batch_images.shape[0]
                break  # Just test first batch


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor."""
    
    def test_preprocessor_creation(self):
        """Test preprocessor creation."""
        preprocessor = ImagePreprocessor(image_size=224)
        assert preprocessor.image_size == 224
        assert len(preprocessor.mean) == 3
        assert len(preprocessor.std) == 3
    
    def test_transforms(self):
        """Test transform creation."""
        train_transform = get_transforms("train")
        val_transform = get_transforms("val")
        
        assert train_transform is not None
        assert val_transform is not None
        
        # Test transform application
        test_image = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        
        train_result = train_transform(test_image)
        val_result = val_transform(test_image)
        
        assert train_result.shape == (3, 224, 224)
        assert val_result.shape == (3, 224, 224)
        assert isinstance(train_result, torch.Tensor)
        assert isinstance(val_result, torch.Tensor)
    
    def test_image_enhancement(self):
        """Test image enhancement."""
        preprocessor = ImagePreprocessor()
        
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        enhanced = preprocessor.enhance_image(test_image)
        
        assert enhanced.size == test_image.size
        assert isinstance(enhanced, Image.Image)


class TestDataUtils:
    """Test cases for data utilities."""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "sample_data"
            
            create_sample_dataset(data_dir, num_samples_per_class=5)
            
            assert data_dir.exists()
            
            # Check that class directories were created
            class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
            assert len(class_dirs) > 0
            
            # Check that images were created
            for class_dir in class_dirs:
                images = list(class_dir.glob("*.jpg"))
                assert len(images) == 5
    
    def test_analyze_dataset(self):
        """Test dataset analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "test_data"
            class_names = ["class1", "class2"]
            
            # Create sample data
            for class_name in class_names:
                class_dir = data_dir / class_name
                class_dir.mkdir(parents=True)
                
                for i in range(3):
                    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                    img.save(class_dir / f"sample_{i}.jpg")
            
            # Analyze dataset
            stats = analyze_dataset(data_dir)
            
            assert stats["total_images"] == 6  # 3 per class
            assert len(stats["classes"]) == 2
            assert "class1" in stats["classes"]
            assert "class2" in stats["classes"]
            assert stats["classes"]["class1"] == 3
            assert stats["classes"]["class2"] == 3


if __name__ == "__main__":
    pytest.main([__file__])
