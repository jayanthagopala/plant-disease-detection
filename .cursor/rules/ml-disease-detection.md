---
type: Auto Attached  
globs: ["**/ml/**", "**/models/**", "**/*disease*", "**/*detection*"]
description: Machine learning guidelines for crop disease detection
---

# ML/Disease Detection Guidelines

## Model Requirements
- Support common Indian crop diseases
- Process images up to 5MB efficiently
- Provide confidence scores
- Return actionable treatment advice in local language

## Image Processing Pipeline
```python
# Standard image preprocessing for crop disease detection
def preprocess_image(image_file):
    image = Image.open(image_file)
    # Resize to model input size (typically 224x224)
    image = image.resize((224, 224))
    # Normalize pixel values
    image_array = np.array(image) / 255.0
    # Handle different image formats (RGBA -> RGB)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    return np.expand_dims(image_array, axis=0)
```

## Model Response Format
```json
{
  "disease": "Late Blight",
  "confidence": 0.87,
  "severity": "moderate",
  "treatment": {
    "english": "Apply copper-based fungicide immediately",
    "hindi": "तुरंत कॉपर आधारित कवकनाशी का प्रयोग करें"
  },
  "prevention": "Remove infected leaves and improve air circulation"
}
```

## Free Model Options
- Use Hugging Face transformers for image classification
- Implement TensorFlow.js for client-side processing
- Consider PlantNet API for plant identification
- Use pre-trained models and fine-tune on crop-specific data
