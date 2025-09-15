"""Image preprocessing utilities for plant disease detection."""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, Optional, List
import cv2


class ImagePreprocessor:
    """Image preprocessing utilities for plant disease detection."""
    
    def __init__(
        self,
        image_size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> None:
        """
        Initialize the image preprocessor.
        
        Args:
            image_size: Target image size
            mean: Normalization mean values
            std: Normalization std values
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
    
    def preprocess_for_training(self) -> transforms.Compose:
        """Get preprocessing transforms for training (with augmentation)."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def preprocess_for_inference(self) -> transforms.Compose:
        """Get preprocessing transforms for inference (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Apply image enhancement techniques.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Attempt to remove background using simple techniques.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Image with background removed (simplified version)
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Create mask for green (plant) regions
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        result = img_array.copy()
        result[mask == 0] = [255, 255, 255]  # Set background to white
        
        return Image.fromarray(result)
    
    def detect_leaf_region(self, image: Image.Image) -> Tuple[int, int, int, int]:
        """
        Detect the main leaf region in the image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Bounding box coordinates (x, y, width, height)
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return x, y, w, h
        else:
            # Return full image if no contour found
            return 0, 0, image.width, image.height


def get_transforms(
    split: str = "train",
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> transforms.Compose:
    """
    Get appropriate transforms for different splits.
    
    Args:
        split: Dataset split ('train', 'val', 'test')
        image_size: Target image size
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Composed transforms
    """
    preprocessor = ImagePreprocessor(image_size, mean, std)
    
    if split == "train":
        return preprocessor.preprocess_for_training()
    else:
        return preprocessor.preprocess_for_inference()


def create_augmentation_pipeline(
    image_size: int = 224,
    augmentation_strength: str = "medium"
) -> transforms.Compose:
    """
    Create a data augmentation pipeline.
    
    Args:
        image_size: Target image size
        augmentation_strength: Strength of augmentation ('light', 'medium', 'heavy')
        
    Returns:
        Composed augmentation transforms
    """
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if augmentation_strength == "light":
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ]
    elif augmentation_strength == "medium":
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1)
        ]
    else:  # heavy
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
                shear=10
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomErasing(p=0.1)
        ]
    
    # Insert augmentation before ToTensor
    all_transforms = base_transforms[:-2] + augmentation_transforms + base_transforms[-2:]
    
    return transforms.Compose(all_transforms)
