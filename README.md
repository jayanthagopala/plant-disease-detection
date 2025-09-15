# 🌱 Plant Disease Detection for Indian Farmers

A deep learning-based plant disease detection system specifically designed for Indian farmers, featuring a user-friendly Streamlit web application for real-time disease identification and treatment recommendations.

## 🎯 Features

- **CNN-based Disease Classification**: Uses transfer learning with ResNet and EfficientNet architectures
- **Streamlit Web Interface**: Mobile-friendly web app for easy access by farmers
- **Multi-crop Support**: Covers major Indian crops (Rice, Wheat, Maize, Tomato, Potato, etc.)
- **Real-time Predictions**: Instant disease detection with confidence scores
- **Treatment Recommendations**: Provides symptoms, treatment, and prevention advice
- **Sample Dataset**: Includes sample data for testing and demonstration
- **Easy Setup**: Simple installation and configuration using `uv`

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- `uv` package manager ([Install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd plant-disease-detection
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

### Running the Application

1. **Start the Streamlit app:**
   ```bash
   python run_app.py
   ```
   
   Or directly with Streamlit:
   ```bash
   uv run streamlit run src/streamlit_app/app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

## 🏗️ Project Structure

```
plant-disease-detection/
├── .cursor/
│   └── rules/                    # Cursor IDE rules
├── src/
│   ├── models/                   # Model definitions
│   │   ├── cnn_model.py         # CNN architecture
│   │   └── disease_classifier.py # Inference wrapper
│   ├── data/                     # Data utilities
│   │   ├── dataset.py           # Dataset classes
│   │   ├── preprocessing.py     # Image preprocessing
│   │   └── utils.py             # Data utilities
│   ├── streamlit_app/           # Streamlit application
│   │   ├── app.py              # Main app
│   │   └── components.py       # Custom components
│   └── train.py                 # Training script
├── data/                        # Dataset directory
├── outputs/                     # Model outputs and results
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── pyproject.toml              # Project configuration
├── run_app.py                  # App launcher
└── README.md                   # This file
```

## 🧠 Model Training

### Training a New Model

1. **Prepare your dataset:**
   - Organize images in folders by disease class
   - Place in `data/plant_diseases/` directory
   - Each class should have its own subfolder

2. **Run training:**
   ```bash
   uv run python src/train.py --data_dir data/plant_diseases --epochs 50 --batch_size 32
   ```

3. **Training options:**
   ```bash
   uv run python src/train.py --help
   ```

### Training Parameters

- `--data_dir`: Path to dataset directory
- `--output_dir`: Path to save model outputs (default: `outputs`)
- `--model_name`: Base model architecture (`resnet18`, `resnet50`, `efficientnet_b0`)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)
- `--create_sample`: Create sample dataset if data directory is empty

### Sample Dataset

Create a sample dataset for testing:
```bash
uv run python src/train.py --create_sample --data_dir data/sample_data
```

## 📱 Using the Web Application

### Image Upload
1. **Upload Image**: Click "Choose an image file" and select a plant image
2. **Use Sample**: Select from pre-loaded sample images
3. **Analyze**: Click "Analyze Image" to get predictions

### Understanding Results
- **Confidence Score**: Higher values indicate more reliable predictions
- **Disease Information**: Symptoms, treatment, and prevention advice
- **Multiple Predictions**: See top 5 predictions with confidence scores

### Supported Image Formats
- JPG, JPEG, PNG, BMP
- Minimum resolution: 224x224 pixels
- Maximum file size: 10MB

## 🔬 Model Architecture

### Base Models
- **ResNet18**: Lightweight, fast inference
- **ResNet50**: Better accuracy, moderate size
- **EfficientNet-B0**: Optimal accuracy/size trade-off

### Custom Classifier Head
- Dropout layers for regularization
- Fully connected layers: 512 → 256 → num_classes
- ReLU activation functions

### Data Augmentation
- Random horizontal/vertical flips
- Random rotation (±15°)
- Color jittering
- Random grayscale conversion

## 📊 Supported Diseases

### Rice
- Rice Blast
- Rice Healthy

### Wheat
- Wheat Rust
- Wheat Healthy

### Maize
- Maize Leaf Blight
- Maize Healthy

### Tomato
- Tomato Early Blight
- Tomato Healthy

### Potato
- Potato Late Blight
- Potato Healthy

### Other Crops
- Sugarcane Red Rot
- Cotton Leaf Spot
- And more...

## 🛠️ Development

### Setting up Development Environment

1. **Install development dependencies:**
   ```bash
   uv sync --extra dev
   ```

2. **Run tests:**
   ```bash
   uv run pytest tests/
   ```

3. **Code formatting:**
   ```bash
   uv run black src/
   uv run isort src/
   ```

4. **Type checking:**
   ```bash
   uv run mypy src/
   ```

### Adding New Diseases

1. **Add disease information** in `src/models/disease_classifier.py`
2. **Update class names** in your dataset
3. **Retrain the model** with new data
4. **Update the web interface** if needed

## 📈 Performance Metrics

The model provides several performance indicators:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate for each class
- **Recall**: Sensitivity for each class
- **F1-Score**: Harmonic mean of precision and recall

## 🌍 Deployment

### Local Deployment
The Streamlit app can be run locally for testing and development.

### Production Deployment
For production deployment, consider:
- Using a cloud platform (AWS, GCP, Azure)
- Implementing proper authentication
- Adding database for user data
- Setting up monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit team for the web application framework
- Agricultural research institutions for disease information
- Indian farming communities for inspiration and feedback

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact your local agricultural extension office
- Check the documentation and FAQ

## 🔮 Future Enhancements

- [ ] Mobile app development
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Integration with weather data
- [ ] Real-time disease monitoring
- [ ] Farmer community features
- [ ] Expert consultation integration

---

**🌱 Built with ❤️ for Indian Farmers**

*This project aims to democratize access to agricultural technology and help farmers make informed decisions about crop health.*
