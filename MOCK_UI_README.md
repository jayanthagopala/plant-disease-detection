# Plant Disease Detection - Mock UI Demo

This is a demonstration version of the Plant Disease Detection app using mock data instead of actual trained models.

## 🚀 Quick Start

1. **Install dependencies** (if not already done):
   ```bash
   uv sync
   ```

2. **Test mock data** (optional):
   ```bash
   uv run python test_mock_data.py
   ```

3. **Run the Streamlit app**:
   ```bash
   uv run python run_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 🌟 Features

### 🔍 Disease Detection Tab
- **Image Upload**: Upload any image to see mock predictions
- **Sample Images**: Use mock sample images for different crop types
- **Real-time Analysis**: Get instant mock predictions with confidence scores
- **Disease Information**: View detailed symptoms, treatment, and prevention info

### 📊 Model Info Tab
- **Model Details**: View mock model specifications
- **Performance Metrics**: See mock accuracy, precision, recall, and F1-score
- **Performance Charts**: Visual representation of model performance

### 📚 Disease Database Tab
- **Crop-specific Diseases**: Browse diseases by crop type
- **Detailed Information**: Symptoms, treatment, and prevention for each disease
- **Expandable Cards**: Easy-to-read disease information

## 🎯 Supported Crops (Mock Data)

- **Rice**: Healthy, Bacterial Leaf Blight, Brown Spot, Blast
- **Wheat**: Healthy, Rust, Powdery Mildew
- **Tomato**: Healthy, Early Blight, Late Blight
- **Potato**: Healthy, Late Blight, Scab

## 🛠️ Technical Details

### Mock Data Features
- **Realistic Predictions**: Generated using probability distributions
- **Confidence Scores**: Random but realistic confidence levels
- **Disease Information**: Comprehensive mock data for each disease
- **Model Performance**: Simulated but realistic performance metrics

### UI Features
- **Mobile-Friendly**: Responsive design for mobile devices
- **Modern Styling**: Beautiful gradients and animations
- **Interactive Charts**: Plotly charts for data visualization
- **Tabbed Interface**: Organized content in easy-to-navigate tabs

## 🎨 UI Components

### Main Interface
- **Header**: Eye-catching title with plant emoji
- **Sidebar**: Settings, model selection, and performance metrics
- **Tabs**: Disease Detection, Model Info, and Disease Database

### Interactive Elements
- **File Uploader**: Drag-and-drop image upload
- **Crop Selector**: Choose crop type for better predictions
- **Confidence Slider**: Adjust prediction threshold
- **Sample Images**: Quick access to mock sample images

### Visualizations
- **Confidence Gauges**: Circular progress indicators
- **Bar Charts**: Disease prediction comparisons
- **Radar Charts**: Multi-dimensional prediction analysis
- **Performance Metrics**: Model accuracy visualization

## 🔧 Customization

### Adding New Diseases
Edit `src/streamlit_app/mock_data.py`:
```python
DISEASE_CATEGORIES["new_crop"] = {
    "disease_name": {
        "name": "Disease Display Name",
        "symptoms": "Disease symptoms description",
        "treatment": "Treatment recommendations",
        "prevention": "Prevention strategies"
    }
}
```

### Modifying Mock Predictions
Adjust the `generate_mock_predictions()` function to change:
- Confidence score ranges
- Number of predictions
- Prediction distribution

### Styling Changes
Update the CSS in `src/streamlit_app/app.py` to modify:
- Colors and gradients
- Layout and spacing
- Mobile responsiveness
- Animation effects

## 📱 Mobile Support

The app is fully responsive and includes:
- **Mobile-optimized headers**: Smaller fonts on mobile devices
- **Touch-friendly buttons**: Larger touch targets
- **Responsive layouts**: Columns stack on small screens
- **Optimized images**: Automatic resizing for mobile

## 🚨 Demo Mode Notice

This is a **demonstration version** using mock data. For production use:
1. Replace mock data with actual trained models
2. Implement real image processing
3. Add database connectivity
4. Include user authentication
5. Add data persistence

## 🛠️ Development

### File Structure
```
src/streamlit_app/
├── app.py              # Main Streamlit application
├── components.py       # Reusable UI components
└── mock_data.py        # Mock data and functions
```

### Key Functions
- `MockDiseaseClassifier`: Mock model class
- `generate_mock_predictions()`: Generate realistic predictions
- `get_mock_disease_info()`: Get disease information
- `create_mock_image_info()`: Process image metadata

## 🎯 Next Steps

1. **Train Real Models**: Replace mock data with actual CNN models
2. **Add Real Data**: Implement actual image processing
3. **Database Integration**: Add persistent storage
4. **User Management**: Add authentication and user accounts
5. **API Integration**: Connect to external agricultural APIs
6. **Multi-language Support**: Add Hindi and other Indian languages

## 📞 Support

For questions or issues with the mock UI:
1. Check the console output for error messages
2. Verify all dependencies are installed with `uv sync`
3. Test mock data with `uv run python test_mock_data.py`
4. Check the Streamlit documentation for UI issues

---

**🌱 Built for Indian Farmers | Powered by AI | Demo Version**
