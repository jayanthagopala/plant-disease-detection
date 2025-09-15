#!/bin/bash

# Plant Disease Detection Setup Script
# This script sets up the development environment

set -e

echo "🌱 Setting up Plant Disease Detection Project..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv is installed"

# Create virtual environment and install dependencies
echo "📦 Installing dependencies..."
uv sync

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/plant_diseases
mkdir -p outputs
mkdir -p notebooks
mkdir -p tests

# Create sample dataset
echo "🌾 Creating sample dataset..."
uv run python -c "
from src.data.utils import create_sample_dataset
from pathlib import Path
create_sample_dataset(Path('data/plant_diseases'), num_samples_per_class=20)
"

echo "✅ Sample dataset created"

# Run basic tests
echo "🧪 Running basic tests..."
uv run python -c "
from src.models.cnn_model import create_model
from src.data.dataset import PlantDiseaseDataset
from pathlib import Path
import torch

# Test model creation
model = create_model(num_classes=10, device='cpu')
print('✅ Model creation test passed')

# Test dataset creation
dataset = PlantDiseaseDataset(
    data_dir=Path('data/plant_diseases'),
    class_names=['rice_blast', 'rice_healthy', 'wheat_rust', 'wheat_healthy', 
                 'maize_leaf_blight', 'maize_healthy', 'tomato_early_blight', 
                 'tomato_healthy', 'potato_late_blight', 'potato_healthy'],
    transform=None
)
print('✅ Dataset creation test passed')
print(f'Dataset size: {len(dataset)} samples')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run the Streamlit app:"
echo "   python run_app.py"
echo ""
echo "3. Or train a model:"
echo "   uv run python src/train.py --create_sample"
echo ""
echo "Happy farming! 🌱"
