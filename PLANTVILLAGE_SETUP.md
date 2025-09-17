# PlantVillage Dataset Setup Guide

This guide will help you download and set up the PlantVillage dataset for training your plant disease detection model.

## 🌱 About PlantVillage Dataset

The PlantVillage dataset contains over 54,000 images of healthy and diseased plant leaves across 14 crop species and 38 disease classes. It's one of the most comprehensive plant disease datasets available.

**Dataset Features:**
- 54,305 images total
- 14 crop species (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- 38 disease classes (including healthy plants)
- High-quality images (256x256 pixels)
- Well-organized structure

## 📋 Prerequisites

1. **Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com)
2. **Kaggle API Token**: 
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Click "Create New API Token"
   - Download `kaggle.json`
   - Place it in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Run the automated setup script
python scripts/download_plantvillage.py
```

### Option 2: Manual Setup

1. **Download from Kaggle**:
   ```bash
   # Install Kaggle API
   pip install kaggle
   
   # Download dataset
   kaggle datasets download -d emmarex/plantdisease -p data/ --unzip
   ```

2. **Organize the data**:
   ```bash
   # The script will automatically organize the data
   # into the expected structure for training
   ```

## 📁 Dataset Structure

After setup, your data directory will look like:

```
data/plant_diseases/
├── apple_scab/
├── apple_black_rot/
├── apple_cedar_rust/
├── apple_healthy/
├── corn_common_rust/
├── corn_healthy/
├── potato_early_blight/
├── potato_late_blight/
├── potato_healthy/
├── tomato_bacterial_spot/
├── tomato_early_blight/
├── tomato_late_blight/
├── tomato_healthy/
└── ... (38 total classes)
```

## 🏋️ Training with PlantVillage Data

Once the dataset is set up, train your model:

```bash
# Basic training
uv run python src/train.py --data_dir data/plant_diseases --epochs 50

# Advanced training with more parameters
uv run python src/train.py \
    --data_dir data/plant_diseases \
    --model_name resnet50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001
```

## 📊 Expected Performance

With the PlantVillage dataset, you can expect:
- **Training accuracy**: 85-95%
- **Validation accuracy**: 80-90%
- **Test accuracy**: 75-85%

The model will be much more accurate than with synthetic data!

## 🔧 Customization

### Filter Specific Crops

If you want to focus on specific crops (e.g., only Indian crops), you can modify the class mapping in `scripts/download_plantvillage.py`:

```python
# Example: Only keep potato and tomato classes
class_mapping = {
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight", 
    "Potato___healthy": "potato_healthy",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___healthy": "tomato_healthy"
}
```

### Data Augmentation

The training script includes data augmentation, but you can customize it in `src/data/preprocessing.py`:

```python
# Add more augmentation techniques
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 🐛 Troubleshooting

### Common Issues

1. **Kaggle API Error**:
   ```
   Error: 401 - Unauthorized
   ```
   - Check your `kaggle.json` file is in the correct location
   - Verify the API token is valid

2. **Download Fails**:
   ```
   Error: Dataset not found
   ```
   - Make sure you have access to the dataset
   - Check your internet connection

3. **Memory Issues**:
   ```
   CUDA out of memory
   ```
   - Reduce batch size: `--batch_size 16`
   - Use smaller model: `--model_name resnet18`

### Getting Help

- Check the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api)
- Visit the [PlantVillage dataset page](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Open an issue in this repository

## 🎯 Next Steps

1. **Download the dataset** using the setup script
2. **Train the model** with real data
3. **Test the Streamlit app** with the new model
4. **Deploy** for Indian farmers

Happy farming! 🌱
