"""Dataset classes for plant disease detection."""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os


class PlantDiseaseDataset(Dataset):
    """Dataset class for plant disease images."""
    
    def __init__(
        self,
        data_dir: Path,
        class_names: List[str],
        transform: Optional[transforms.Compose] = None,
        split: str = "train"
    ) -> None:
        """
        Initialize the plant disease dataset.
        
        Args:
            data_dir: Directory containing the dataset
            class_names: List of class names
            transform: Image transformations to apply
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.transform = transform
        self.split = split
        
        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load image paths and corresponding labels."""
        samples = []
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob("*.jpg"):
                samples.append((img_path, self.class_to_idx[class_name]))
            for img_path in class_dir.glob("*.png"):
                samples.append((img_path, self.class_to_idx[class_name]))
            for img_path in class_dir.glob("*.jpeg"):
                samples.append((img_path, self.class_to_idx[class_name]))
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {}
        for _, label in self.samples:
            class_name = self.class_names[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


def create_data_loaders(
    data_dir: Path,
    class_names: List[str],
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the dataset
        class_names: List of class names
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        num_workers: Number of worker processes
        image_size: Size to resize images to
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = PlantDiseaseDataset(
        data_dir=data_dir,
        class_names=class_names,
        transform=train_transform
    )
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, temp_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size + test_size]
    )
    
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size]
    )
    
    # Update transforms for validation and test datasets
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names


def create_sample_dataset(
    output_dir: Path,
    num_samples_per_class: int = 10
) -> None:
    """
    Create a sample dataset for testing purposes.
    
    Args:
        output_dir: Directory to create the sample dataset
        num_samples_per_class: Number of sample images per class
    """
    # Sample class names for Indian crops
    class_names = [
        "rice_blast",
        "rice_healthy", 
        "wheat_rust",
        "wheat_healthy",
        "maize_leaf_blight",
        "maize_healthy",
        "tomato_early_blight",
        "tomato_healthy",
        "potato_late_blight",
        "potato_healthy"
    ]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in class_names:
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create placeholder images (in a real scenario, you'd have actual images)
        for i in range(num_samples_per_class):
            # Create a simple colored image as placeholder
            img = Image.new('RGB', (224, 224), 
                          color=(100 + i * 10, 150 + i * 5, 200 - i * 8))
            img.save(class_dir / f"sample_{i:03d}.jpg")
    
    print(f"Sample dataset created at {output_dir}")
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(class_names) * num_samples_per_class}")
