"""Utility functions for data handling."""

import os
import requests
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from PIL import Image
import numpy as np


def download_sample_data(
    output_dir: Path,
    dataset_url: Optional[str] = None
) -> None:
    """
    Download sample plant disease dataset.
    
    Args:
        output_dir: Directory to download the dataset
        dataset_url: URL to download dataset from (optional)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_url:
        # Download from provided URL
        print(f"Downloading dataset from {dataset_url}")
        response = requests.get(dataset_url)
        
        zip_path = output_dir / "dataset.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Remove zip file
        zip_path.unlink()
    else:
        # Create sample dataset locally
        create_sample_dataset(output_dir)


def create_sample_dataset(
    output_dir: Path,
    num_samples_per_class: int = 20
) -> None:
    """
    Create a sample dataset for testing purposes.
    
    Args:
        output_dir: Directory to create the sample dataset
        num_samples_per_class: Number of sample images per class
    """
    # Sample class names for Indian crops and their diseases
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
        "potato_healthy",
        "sugarcane_red_rot",
        "sugarcane_healthy",
        "cotton_leaf_spot",
        "cotton_healthy"
    ]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in class_names:
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create placeholder images with different patterns
        for i in range(num_samples_per_class):
            # Create images with different colors and patterns based on class
            if "healthy" in class_name:
                # Green healthy plant colors
                base_color = (34, 139, 34)  # Forest green
                variation = 30
            else:
                # Diseased plant colors (more brown/yellow)
                base_color = (139, 69, 19)  # Saddle brown
                variation = 40
            
            # Add some variation to the base color
            r = max(0, min(255, base_color[0] + np.random.randint(-variation, variation)))
            g = max(0, min(255, base_color[1] + np.random.randint(-variation, variation)))
            b = max(0, min(255, base_color[2] + np.random.randint(-variation, variation)))
            
            # Create image with some texture
            img = Image.new('RGB', (224, 224), (r, g, b))
            
            # Add some noise/texture
            img_array = np.array(img)
            noise = np.random.randint(-20, 20, img_array.shape)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Add some circular patterns to simulate leaf structures
            center_x, center_y = 112, 112
            for radius in range(20, 100, 20):
                for angle in range(0, 360, 30):
                    x = int(center_x + radius * np.cos(np.radians(angle)))
                    y = int(center_y + radius * np.sin(np.radians(angle)))
                    if 0 <= x < 224 and 0 <= y < 224:
                        img_array[y-2:y+2, x-2:x+2] = [r//2, g//2, b//2]
            
            img = Image.fromarray(img_array)
            img.save(class_dir / f"sample_{i:03d}.jpg")
    
    print(f"Sample dataset created at {output_dir}")
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(class_names) * num_samples_per_class}")


def analyze_dataset(data_dir: Path) -> Dict[str, any]:
    """
    Analyze the dataset and return statistics.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "total_images": 0,
        "classes": {},
        "image_formats": {},
        "image_sizes": [],
        "class_distribution": {}
    }
    
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        class_images = []
        
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    with Image.open(img_file) as img:
                        class_images.append(img_file)
                        stats["total_images"] += 1
                        
                        # Track image format
                        format_name = img.format or "unknown"
                        stats["image_formats"][format_name] = stats["image_formats"].get(format_name, 0) + 1
                        
                        # Track image size
                        stats["image_sizes"].append(img.size)
                        
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        
        stats["classes"][class_name] = len(class_images)
        stats["class_distribution"][class_name] = len(class_images)
    
    # Calculate average image size
    if stats["image_sizes"]:
        avg_width = sum(size[0] for size in stats["image_sizes"]) / len(stats["image_sizes"])
        avg_height = sum(size[1] for size in stats["image_sizes"]) / len(stats["image_sizes"])
        stats["average_size"] = (int(avg_width), int(avg_height))
    
    return stats


def create_dataset_info_file(data_dir: Path, output_file: Path) -> None:
    """
    Create a CSV file with dataset information.
    
    Args:
        data_dir: Directory containing the dataset
        output_file: Path to save the CSV file
    """
    data = []
    
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    with Image.open(img_file) as img:
                        data.append({
                            "file_path": str(img_file),
                            "class_name": class_name,
                            "width": img.size[0],
                            "height": img.size[1],
                            "format": img.format or "unknown",
                            "mode": img.mode
                        })
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Dataset info saved to {output_file}")


def validate_dataset(data_dir: Path) -> List[str]:
    """
    Validate the dataset and return any issues found.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        List of validation issues
    """
    issues = []
    
    if not data_dir.exists():
        issues.append(f"Data directory {data_dir} does not exist")
        return issues
    
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        issues.append("No class directories found")
        return issues
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = []
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(class_dir.glob(f"*{ext}")))
        
        if not image_files:
            issues.append(f"No images found in class directory: {class_name}")
            continue
        
        # Check for corrupted images
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    img.verify()
            except Exception as e:
                issues.append(f"Corrupted image {img_file}: {e}")
    
    return issues
