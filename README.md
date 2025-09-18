# 🌱 Plant Disease Detection for Indian Farmers

A deep learning-based plant disease detection system specifically designed for Indian farmers, featuring a user-friendly Streamlit web application for real-time disease identification and treatment recommendations.

## 🎯 Features

- **CNN-based Disease Classification**: Uses transfer learning with ResNet and EfficientNet architectures
- **Streamlit Web Interface**: Mobile-friendly web app for easy access by farmers
- **Multi-crop Support**: Covers major Indian crops (Rice, Wheat, Maize, Tomato, Potato, etc.)
- **Real-time Predictions**: Instant disease detection with confidence scores
- **Treatment Recommendations**: Provides symptoms, treatment, and prevention advice
- **Real-time Weather Data**: Live weather information and forecasts for farming decisions
- **Weather Alerts**: Intelligent alerts for extreme weather conditions
- **Crop-specific Recommendations**: Weather-based advice tailored to different crops
- **Multilingual Support**: English, Hindi, and Kannada language support
- **Market Price Tracking**: Real-time crop price information and trends
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
   
   This will automatically:
   - Create a virtual environment
   - Install all dependencies from `pyproject.toml`
   - Install development dependencies if needed

3. **Activate the virtual environment (optional):**
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

## 📚 Documentation

- **[Product Requirements Document](docs/PRD.md)** - Detailed project specifications and requirements
- **[Weather API Guide](docs/WEATHER_API_GUIDE.md)** - Comprehensive weather API documentation
- **[PlantVillage Setup](docs/PLANTVILLAGE_SETUP.md)** - Dataset setup and configuration guide

## 🏗️ Project Structure

```
plant-disease-detection/
├── .cursor/
│   └── rules/                    # Cursor IDE rules
├── docs/                        # Documentation
│   ├── PRD.md                   # Product Requirements Document
│   ├── WEATHER_API_GUIDE.md     # Weather API documentation
│   └── PLANTVILLAGE_SETUP.md    # Dataset setup guide
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
│   │   ├── components.py       # Custom components
│   │   ├── weather_service.py  # Weather API service
│   │   ├── market_service.py   # Market price service
│   │   └── translations.py     # Multilingual support
│   └── train.py                 # Training script
├── data/                        # Dataset directory
├── outputs/                     # Model outputs and results
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
│   ├── test_data.py            # Data utilities tests
│   ├── test_model.py           # Model testing
│   ├── test_models.py          # Model architecture tests
│   └── test_new_features.py    # New features integration tests
├── pyproject.toml              # Project configuration and dependencies
├── run_app.py                  # App launcher
├── run_tests.py                # Test runner script
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

### Dependency Management

This project uses `pyproject.toml` for dependency management with `uv`. No `requirements.txt` file is needed.

**Main dependencies** (installed with `uv sync`):
- PyTorch & Torchvision for deep learning
- Streamlit for web interface
- OpenCV, PIL for image processing
- Pandas, NumPy for data handling
- Requests for API calls
- Plotly for visualizations

**Development dependencies** (installed with `uv sync --extra dev`):
- pytest for testing
- black, isort for code formatting
- mypy for type checking
- flake8 for linting

### Setting up Development Environment

1. **Install development dependencies:**
   ```bash
   uv sync --extra dev
   ```

2. **Run tests:**
   ```bash
   # Run all tests using the test runner
   uv run python run_tests.py
   
   # Run specific test files
   uv run python tests/test_model.py
   uv run python tests/test_new_features.py
   
   # Run with pytest (if installed)
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
- [x] Multi-language support (Hindi, Kannada) - **Implemented**
- [x] Integration with weather data - **Implemented**
- [ ] Real-time disease monitoring
- [ ] Farmer community features
- [ ] Expert consultation integration
- [ ] Soil moisture integration
- [ ] Pest prediction based on weather
- [ ] Automated irrigation scheduling

---

**🌱 Built with ❤️ for Indian Farmers**

*This project aims to democratize access to agricultural technology and help farmers make informed decisions about crop health.*
