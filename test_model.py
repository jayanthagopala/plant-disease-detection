#!/usr/bin/env python3
"""Test the trained plant disease detection model."""

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.disease_classifier import DiseaseClassifier


def test_model():
    """Test the trained model with sample images."""
    print("🌱 Testing Plant Disease Detection Model")
    print("=" * 50)
    
    # Load the trained model
    model_path = Path("outputs/best_model.pth")
    if not model_path.exists():
        print("❌ Model file not found. Please train a model first.")
        return
    
    print("📦 Loading trained model...")
    classifier = DiseaseClassifier(
        model_path=model_path,
        device="cpu"
    )
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Number of classes: {len(classifier.class_names)}")
    print(f"🏷️  Classes: {classifier.class_names}")
    
    # Test with sample images
    data_dir = Path("data/plant_diseases")
    if not data_dir.exists():
        print("❌ Sample data not found. Please create sample data first.")
        return
    
    print("\n🔍 Testing with sample images...")
    print("-" * 30)
    
    # Test a few images from each class
    test_count = 0
    correct_predictions = 0
    
    for class_name in classifier.class_names[:5]:  # Test first 5 classes
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
            
        # Get first image from this class
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
        if not image_files:
            continue
            
        test_image = image_files[0]
        print(f"\n📸 Testing image: {test_image.name}")
        print(f"🏷️  True class: {class_name}")
        
        try:
            # Load and predict
            image = Image.open(test_image)
            predictions = classifier.predict(image, top_k=3)
            
            # Display top 3 predictions
            print("🔮 Predictions:")
            for i, pred in enumerate(predictions, 1):
                confidence = pred['confidence']
                pred_class = pred['class_name']
                print(f"   {i}. {pred_class} ({confidence:.2%})")
            
            # Check if top prediction is correct
            top_prediction = predictions[0]['class_name']
            if top_prediction == class_name:
                correct_predictions += 1
                print("✅ Correct prediction!")
            else:
                print("❌ Incorrect prediction")
                
            test_count += 1
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
    
    # Calculate accuracy
    if test_count > 0:
        accuracy = correct_predictions / test_count
        print(f"\n📊 Test Results:")
        print(f"   Total images tested: {test_count}")
        print(f"   Correct predictions: {correct_predictions}")
        print(f"   Accuracy: {accuracy:.2%}")
    else:
        print("❌ No images were tested successfully.")
    
    print("\n🎯 Model testing completed!")


def test_single_image(image_path: str):
    """Test the model with a single image."""
    print(f"🔍 Testing single image: {image_path}")
    
    model_path = Path("outputs/best_model.pth")
    if not model_path.exists():
        print("❌ Model file not found. Please train a model first.")
        return
    
    # Load model
    classifier = DiseaseClassifier(model_path=model_path, device="cpu")
    
    # Load and predict
    try:
        image = Image.open(image_path)
        predictions = classifier.predict(image, top_k=5)
        
        print("\n🔮 Predictions:")
        for i, pred in enumerate(predictions, 1):
            confidence = pred['confidence']
            pred_class = pred['class_name']
            print(f"   {i}. {pred_class} ({confidence:.2%})")
        
        # Get disease information
        top_prediction = predictions[0]['class_name']
        disease_info = classifier.get_disease_info(top_prediction)
        
        print(f"\n📋 Disease Information:")
        print(f"   Name: {disease_info['name']}")
        print(f"   Symptoms: {disease_info['symptoms']}")
        print(f"   Treatment: {disease_info['treatment']}")
        print(f"   Prevention: {disease_info['prevention']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test single image
        image_path = sys.argv[1]
        test_single_image(image_path)
    else:
        # Test with sample images
        test_model()
