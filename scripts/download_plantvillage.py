#!/usr/bin/env python3
"""Download and set up the PlantVillage dataset from Kaggle."""

import os
import zipfile
import shutil
from pathlib import Path
import requests
import subprocess
import sys


def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up."""
    kaggle_dir = Path.home() / ".kaggle"
    credentials_file = kaggle_dir / "kaggle.json"
    
    if not credentials_file.exists():
        print("❌ Kaggle API credentials not found!")
        print("\n📋 To download the dataset, you need to:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/ directory")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle credentials found!")
    return True


def install_kaggle_api():
    """Install Kaggle API if not already installed."""
    try:
        import kaggle
        print("✅ Kaggle API already installed")
        return True
    except ImportError:
        print("📦 Installing Kaggle API...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            print("✅ Kaggle API installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Kaggle API")
            return False


def download_plantvillage_dataset():
    """Download the PlantVillage dataset from Kaggle."""
    print("🌱 Downloading PlantVillage dataset...")
    
    try:
        # Download the dataset
        subprocess.check_call([
            "kaggle", "datasets", "download", 
            "-d", "emmarex/plantdisease",
            "-p", "data/",
            "--unzip"
        ])
        print("✅ Dataset downloaded and extracted successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        return False


def organize_dataset():
    """Organize the downloaded dataset into the expected structure."""
    print("📁 Organizing dataset structure...")
    
    data_dir = Path("data")
    plantvillage_dir = data_dir / "PlantVillage"
    
    if not plantvillage_dir.exists():
        print("❌ PlantVillage directory not found!")
        return False
    
    # Create the expected directory structure
    target_dir = data_dir / "plant_diseases"
    target_dir.mkdir(exist_ok=True)
    
    # Find all class directories
    class_dirs = [d for d in plantvillage_dir.iterdir() if d.is_dir()]
    
    print(f"📊 Found {len(class_dirs)} classes")
    
    # Copy and rename classes to our expected format
    class_mapping = {
        "Apple___Apple_scab": "apple_scab",
        "Apple___Black_rot": "apple_black_rot", 
        "Apple___Cedar_apple_rust": "apple_cedar_rust",
        "Apple___healthy": "apple_healthy",
        "Blueberry___healthy": "blueberry_healthy",
        "Cherry_(including_sour)___Powdery_mildew": "cherry_powdery_mildew",
        "Cherry_(including_sour)___healthy": "cherry_healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn_cercospora_spot",
        "Corn_(maize)___Common_rust_": "corn_common_rust",
        "Corn_(maize)___Northern_Leaf_Blight": "corn_northern_blight",
        "Corn_(maize)___healthy": "corn_healthy",
        "Grape___Black_rot": "grape_black_rot",
        "Grape___Esca_(Black_Measles)": "grape_esca",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape_leaf_blight",
        "Grape___healthy": "grape_healthy",
        "Orange___Haunglongbing_(Citrus_greening)": "orange_huanglongbing",
        "Peach___Bacterial_spot": "peach_bacterial_spot",
        "Peach___healthy": "peach_healthy",
        "Pepper,_bell___Bacterial_spot": "pepper_bacterial_spot",
        "Pepper,_bell___healthy": "pepper_healthy",
        "Potato___Early_blight": "potato_early_blight",
        "Potato___Late_blight": "potato_late_blight",
        "Potato___healthy": "potato_healthy",
        "Raspberry___healthy": "raspberry_healthy",
        "Soybean___healthy": "soybean_healthy",
        "Squash___Powdery_mildew": "squash_powdery_mildew",
        "Strawberry___Leaf_scorch": "strawberry_leaf_scorch",
        "Strawberry___healthy": "strawberry_healthy",
        "Tomato___Bacterial_spot": "tomato_bacterial_spot",
        "Tomato___Early_blight": "tomato_early_blight",
        "Tomato___Late_blight": "tomato_late_blight",
        "Tomato___Leaf_Mold": "tomato_leaf_mold",
        "Tomato___Septoria_leaf_spot": "tomato_septoria_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
        "Tomato___Target_Spot": "tomato_target_spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_curl",
        "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus",
        "Tomato___healthy": "tomato_healthy"
    }
    
    copied_classes = 0
    total_images = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        if class_name in class_mapping:
            new_class_name = class_mapping[class_name]
            target_class_dir = target_dir / new_class_name
            target_class_dir.mkdir(exist_ok=True)
            
            # Copy images
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
            for img_file in image_files:
                shutil.copy2(img_file, target_class_dir)
                total_images += 1
            
            copied_classes += 1
            print(f"✅ {class_name} -> {new_class_name} ({len(image_files)} images)")
    
    print(f"\n📊 Dataset organization complete!")
    print(f"   Classes copied: {copied_classes}")
    print(f"   Total images: {total_images}")
    
    return True


def create_dataset_info():
    """Create a dataset information file."""
    data_dir = Path("data/plant_diseases")
    
    if not data_dir.exists():
        return False
    
    # Count images per class
    class_info = {}
    total_images = 0
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            image_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.JPG")))
            class_info[class_dir.name] = image_count
            total_images += image_count
    
    # Create info file
    info_file = data_dir / "dataset_info.txt"
    with open(info_file, 'w') as f:
        f.write("PlantVillage Dataset Information\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Number of classes: {len(class_info)}\n\n")
        f.write("Class distribution:\n")
        f.write("-" * 20 + "\n")
        
        for class_name, count in sorted(class_info.items()):
            f.write(f"{class_name}: {count} images\n")
    
    print(f"📄 Dataset info saved to {info_file}")
    return True


def main():
    """Main function to download and set up the PlantVillage dataset."""
    print("🌱 PlantVillage Dataset Setup")
    print("=" * 40)
    
    # Check if already downloaded
    if Path("data/plant_diseases").exists() and any(Path("data/plant_diseases").iterdir()):
        print("⚠️  Dataset already exists!")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("✅ Using existing dataset")
            return True
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        return False
    
    # Install Kaggle API
    if not install_kaggle_api():
        return False
    
    # Download dataset
    if not download_plantvillage_dataset():
        return False
    
    # Organize dataset
    if not organize_dataset():
        return False
    
    # Create dataset info
    create_dataset_info()
    
    print("\n🎉 PlantVillage dataset setup complete!")
    print("📁 Dataset location: data/plant_diseases/")
    print("🚀 You can now train the model with real data!")
    
    return True


if __name__ == "__main__":
    main()
