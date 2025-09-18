"""Main Streamlit application for plant disease detection."""

import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import real model and mock data for fallback
try:
    from models.disease_classifier import DiseaseClassifier
    REAL_MODEL_AVAILABLE = True
except ImportError:
    REAL_MODEL_AVAILABLE = False

from mock_data import (
    MockDiseaseClassifier, 
    get_mock_disease_info, 
    create_mock_image_info,
    get_mock_model_performance,
    get_supported_crops,
    create_mock_sample_images,
    DISEASE_CATEGORIES
)

# Import new services
from translations import get_translation, create_language_selector, get_current_language
from weather_service import WeatherService, create_mock_weather_data, get_weather_recommendations
from market_service import MarketPriceService, create_mock_market_data, get_market_insights


@st.cache_resource
def load_disease_classifier():
    """Load the disease classifier model."""
    if REAL_MODEL_AVAILABLE:
        try:
            model_path = Path("outputs/best_model.pth")
            if model_path.exists():
                # Determine device
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                classifier = DiseaseClassifier(model_path=model_path, device=device)
                return classifier, True
        except Exception as e:
            st.error(f"Error loading real model: {e}")
    
    # Fallback to mock data
    return MockDiseaseClassifier(), False


# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 300;
    }
    
    /* Prediction boxes */
    .prediction-box {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .prediction-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Disease info boxes */
    .disease-info {
        background: linear-gradient(135deg, #fff8dc 0%, #fffacd 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ffa500;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
    }
    
    /* Demo banner */
    .demo-banner {
        background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #0277bd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #2E8B57;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8fff8;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #228B22 0%, #1e7e1e 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: white;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1.2rem;
        }
        .prediction-box, .disease-info {
            padding: 1rem;
        }
        .metric-card {
            padding: 1rem;
            margin: 0.25rem;
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
    }
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
    }
    .stWarning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #2E8B57;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #228B22;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str, device: str = "cpu") -> MockDiseaseClassifier:
    """Load the mock model with caching."""
    try:
        classifier = MockDiseaseClassifier(
            model_path=model_path,
            device=device
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_sample_data_if_needed(data_dir: Path) -> None:
    """Create sample data if the data directory is empty."""
    # For mock version, we don't need to create actual sample data
    # Just return success message
    st.success("Using mock sample data for demonstration!")


def display_prediction_results(
    predictions: List[Dict[str, float]], 
    classifier: MockDiseaseClassifier
) -> None:
    """Display prediction results in a nice format."""
    st.markdown("### 🔍 Prediction Results")
    
    # Top prediction
    top_pred = predictions[0]
    confidence = top_pred['confidence']
    
    # Color code based on confidence
    if confidence > 0.8:
        confidence_color = "green"
        confidence_emoji = "🟢"
    elif confidence > 0.6:
        confidence_color = "orange"
        confidence_emoji = "🟡"
    else:
        confidence_color = "red"
        confidence_emoji = "🔴"
    
    # Display top prediction
    st.markdown(f"""
    <div class="prediction-box">
        <h3>{confidence_emoji} {top_pred['class_name'].replace('_', ' ').title()}</h3>
        <p><strong>Confidence:</strong> <span style="color: {confidence_color};">{confidence:.2%}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display all predictions
    st.markdown("#### All Predictions")
    
    # Create a DataFrame for better display
    import pandas as pd
    pred_data = []
    for i, pred in enumerate(predictions):
        pred_data.append({
            'Rank': i + 1,
            'Disease': pred['class_name'].replace('_', ' ').title(),
            'Confidence': f"{pred['confidence']:.2%}",
            'Confidence_Value': pred['confidence']
        })
    
    df = pd.DataFrame(pred_data)
    
    # Display as table
    st.dataframe(df, use_container_width=True)
    
    # Create confidence bar chart
    fig = px.bar(
        df, 
        x='Confidence_Value', 
        y='Disease',
        orientation='h',
        title="Prediction Confidence Scores",
        color='Confidence_Value',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Disease",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display disease information
    disease_info = classifier.get_disease_info(top_pred['class_name'])
    st.markdown("#### 📋 Disease Information")
    st.markdown(f"""
    <div class="disease-info">
        <h4>{disease_info['name']}</h4>
        <p><strong>Symptoms:</strong> {disease_info['symptoms']}</p>
        <p><strong>Treatment:</strong> {disease_info['treatment']}</p>
        <p><strong>Prevention:</strong> {disease_info['prevention']}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Initialize language
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    
    current_lang = get_current_language()
    
    # Load the disease classifier
    classifier, is_real_model = load_disease_classifier()
    
    # Header
    st.markdown(f'<h1 class="main-header">{get_translation("app_title", current_lang)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{get_translation("app_subtitle", current_lang)}</p>', unsafe_allow_html=True)
    
    # Model status
    if is_real_model:
        st.success("✅ Using trained ResNet-18 model (88.86% accuracy)")
    else:
        st.warning("⚠️ Using mock model for demonstration")
    
    # Language selector
    st.markdown("---")
    create_language_selector()
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title(get_translation("settings", current_lang))
    
    # Model info
    if is_real_model:
        st.sidebar.success("🤖 Real Model Active")
        st.sidebar.info(f"📊 Accuracy: 88.86%")
        st.sidebar.info(f"🏷️ Classes: {len(classifier.class_names)}")
    else:
        st.sidebar.warning("🎭 Mock Model Active")
    
    # Device selection
    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    if torch.backends.mps.is_available():
        device_options.append("mps")
    
    device = st.sidebar.selectbox(
        get_translation("device", current_lang),
        device_options,
        help="Select the device for inference"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        get_translation("confidence_threshold", current_lang),
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for predictions"
    )
    
    # Crop type selection
    crop_type = st.sidebar.selectbox(
        get_translation("crop_type", current_lang),
        get_supported_crops(),
        help="Select the crop type for better predictions"
    )
    
    # Location selection for weather
    st.sidebar.markdown("### 🌍 Location")
    city = st.sidebar.text_input("City", value="Mysore", help="Enter your city for weather data")
    state = st.sidebar.text_input("State", value="Karnataka", help="Enter your state")
    
    # Set the classifier in session state
    st.session_state.classifier = classifier
    st.session_state.is_real_model = is_real_model
    
    # Model performance metrics
    if st.sidebar.checkbox("Show Model Performance", value=True):
        st.sidebar.markdown("#### 📊 Model Performance")
        performance = get_mock_model_performance()
        st.sidebar.metric("Accuracy", f"{performance['accuracy']:.1%}")
        st.sidebar.metric("Precision", f"{performance['precision']:.1%}")
        st.sidebar.metric("Recall", f"{performance['recall']:.1%}")
        st.sidebar.metric("F1-Score", f"{performance['f1_score']:.1%}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        get_translation("disease_detection", current_lang), 
        get_translation("model_info", current_lang), 
        get_translation("disease_database", current_lang),
        get_translation("weather_forecast", current_lang),
        get_translation("market_prices", current_lang)
    ])
    
    with tab1:
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Image")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload an image of a plant leaf to detect diseases"
            )
            
            # Sample images
            st.markdown("#### 📸 Sample Images")
            create_sample_data_if_needed(Path("data/plant_diseases"))
            
            # Mock sample images
            sample_images = create_mock_sample_images()
            if crop_type in sample_images:
                diseases = sample_images[crop_type]
                if diseases:
                    selected_disease = st.selectbox("Select sample disease:", diseases)
                    
                    if st.button("Use Sample Image"):
                        # Create a mock image for demonstration
                        mock_image = Image.new('RGB', (224, 224), color=(34, 139, 34))  # Green background
                        st.session_state.mock_sample_image = mock_image
                        uploaded_file = mock_image
        
        with col2:
            st.markdown("### 🔍 Analysis")
            
            if uploaded_file is not None:
                # Display uploaded image
                if isinstance(uploaded_file, str):
                    image = Image.open(uploaded_file)
                else:
                    # Convert UploadedFile to PIL Image
                    image = Image.open(uploaded_file)
                    
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Image info
                image_info = create_mock_image_info(image)
                st.markdown("#### Image Information")
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("Width", f"{image_info['width']}px")
                    st.metric("Height", f"{image_info['height']}px")
                with col_info2:
                    st.metric("Format", image_info['format'])
                    st.metric("Mode", image_info['mode'])
                
                # Predict button
                if st.button("🔍 Analyze Image", type="primary"):
                    if st.session_state.classifier is None:
                        st.error("Please load a model first!")
                    else:
                        with st.spinner("Analyzing image..."):
                            try:
                                # Get predictions from the classifier
                                predictions = st.session_state.classifier.predict(image, top_k=5)
                                
                                # Filter by confidence threshold
                                filtered_predictions = [
                                    p for p in predictions 
                                    if p['confidence'] >= confidence_threshold
                                ]
                                
                                if filtered_predictions:
                                    display_prediction_results(
                                        filtered_predictions, 
                                        st.session_state.classifier
                                    )
                                else:
                                    st.warning(f"No predictions above confidence threshold of {confidence_threshold:.0%}")
                                    
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")
                                st.error(f"Error details: {str(e)}")
            else:
                st.info("Please upload an image or select a sample image to get started.")
    
    with tab2:
        st.markdown("### 🤖 Model Information")
        
        if st.session_state.classifier:
            if st.session_state.is_real_model:
                # Real model information
                st.success("✅ Using Trained ResNet-18 Model")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Model Details")
                    st.metric("Model Type", "ResNet-18")
                    st.metric("Input Size", "224x224")
                    st.metric("Number of Classes", len(classifier.class_names))
                    st.metric("Device", device)
                
                with col2:
                    st.markdown("#### Performance Metrics")
                    st.metric("Test Accuracy", "88.86%")
                    st.metric("Best Validation Accuracy", "89.18%")
                    st.metric("Training Epochs", "5")
                    st.metric("Model Status", "Trained")
                
                # Display class names
                st.markdown("#### Supported Classes")
                class_cols = st.columns(3)
                for i, class_name in enumerate(classifier.class_names):
                    with class_cols[i % 3]:
                        st.text(f"• {class_name.replace('_', ' ').title()}")
            else:
                # Mock model information
                model_info = st.session_state.classifier.get_model_info()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Model Details")
                    st.metric("Model Type", model_info['model_type'])
                    st.metric("Input Size", f"{model_info['input_size'][0]}x{model_info['input_size'][1]}")
                    st.metric("Number of Classes", model_info['num_classes'])
                    st.metric("Model Size", f"{model_info['model_size_mb']:.1f} MB")
                
                with col2:
                    st.markdown("#### Performance Metrics")
                    st.metric("Training Accuracy", f"{model_info['training_accuracy']:.1%}")
                    st.metric("Validation Accuracy", f"{model_info['validation_accuracy']:.1%}")
                
                # Performance chart
                performance = get_mock_model_performance()
                fig = px.bar(
                    x=list(performance.keys()),
                    y=list(performance.values()),
                    title="Model Performance Metrics",
                    color=list(performance.values()),
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown(f"### {get_translation('disease_database', current_lang)}")
        
        # Show diseases for selected crop
        if crop_type in DISEASE_CATEGORIES:
            st.markdown(f"#### {crop_type.title()} {get_translation('diseases', current_lang)}")
            
            for disease, info in DISEASE_CATEGORIES[crop_type].items():
                with st.expander(f"🌿 {info['name']}", expanded=False):
                    st.markdown(f"**{get_translation('symptoms', current_lang)}:** {info['symptoms']}")
                    st.markdown(f"**{get_translation('treatment', current_lang)}:** {info['treatment']}")
                    st.markdown(f"**{get_translation('prevention', current_lang)}:** {info['prevention']}")
        else:
            st.info("Select a crop type from the sidebar to view disease information.")
    
    with tab4:
        st.markdown(f"### {get_translation('weather_forecast', current_lang)}")
        
        # Initialize weather service
        weather_service = WeatherService()
        
        # Get weather data
        with st.spinner("Fetching weather data..."):
            # Try to get real weather data, fallback to mock
            try:
                coords = weather_service.get_coordinates(city, state)
                if coords:
                    current_weather = weather_service.get_current_weather(coords["latitude"], coords["longitude"])
                    forecast = weather_service.get_forecast(coords["latitude"], coords["longitude"])
                    alerts = weather_service.get_weather_alerts(coords["latitude"], coords["longitude"])
                else:
                    current_weather = None
                    forecast = None
                    alerts = []
            except:
                # Use mock data
                weather_data = create_mock_weather_data()
                current_weather = weather_data["current"]
                forecast = weather_data["forecast"]
                alerts = weather_data["alerts"]
        
        # Display current weather
        if current_weather:
            st.markdown(f"#### {get_translation('current_weather', current_lang)}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    get_translation('temperature', current_lang),
                    f"{current_weather['temperature']:.1f}°C"
                )
            
            with col2:
                st.metric(
                    get_translation('humidity', current_lang),
                    f"{current_weather['humidity']:.0f}%"
                )
            
            with col3:
                st.metric(
                    get_translation('rainfall', current_lang),
                    f"{current_weather['precipitation']:.1f}mm"
                )
            
            with col4:
                st.metric(
                    get_translation('wind_speed', current_lang),
                    f"{current_weather['wind_speed']:.1f} km/h"
                )
        
        # Display weather alerts
        if alerts:
            st.markdown(f"#### {get_translation('weather_alert', current_lang)}")
            for alert in alerts:
                st.warning(f"{alert['icon']} {alert['message']}")
        
        # Display forecast
        if forecast:
            st.markdown("#### 7-Day Forecast")
            
            # Create forecast chart
            dates = [day["date"] for day in forecast]
            max_temps = [day["max_temp"] for day in forecast]
            min_temps = [day["min_temp"] for day in forecast]
            precipitation = [day["precipitation"] for day in forecast]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Temperature Forecast", "Precipitation Forecast"),
                vertical_spacing=0.1
            )
            
            # Temperature chart
            fig.add_trace(
                go.Scatter(x=dates, y=max_temps, name="Max Temp", line=dict(color="red")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=min_temps, name="Min Temp", line=dict(color="blue")),
                row=1, col=1
            )
            
            # Precipitation chart
            fig.add_trace(
                go.Bar(x=dates, y=precipitation, name="Precipitation", marker_color="lightblue"),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Weather recommendations
        if current_weather:
            recommendations = get_weather_recommendations({"current": current_weather, "forecast": forecast}, crop_type)
            if recommendations:
                st.markdown(f"#### {get_translation('recommendations', current_lang)}")
                for rec in recommendations:
                    if rec["priority"] == "high":
                        st.error(f"{rec['icon']} {rec['message']}")
                    elif rec["priority"] == "medium":
                        st.warning(f"{rec['icon']} {rec['message']}")
                    else:
                        st.info(f"{rec['icon']} {rec['message']}")
    
    with tab5:
        st.markdown(f"### {get_translation('market_prices', current_lang)}")
        
        # Initialize market service
        market_service = MarketPriceService()
        
        # Get market data
        with st.spinner("Fetching market data..."):
            try:
                prices = market_service.get_crop_prices(state, city)
                if not prices:
                    prices = create_mock_market_data()["prices"]
            except:
                prices = create_mock_market_data()["prices"]
        
        # Display current prices
        if prices:
            st.markdown(f"#### {get_translation('crop_prices', current_lang)}")
            
            # Filter prices for selected crop
            crop_prices = [p for p in prices if crop_type.lower() in p.get("commodity", "").lower()]
            
            if crop_prices:
                # Display prices in a table
                import pandas as pd
                df = pd.DataFrame(crop_prices)
                st.dataframe(df[["commodity", "variety", "market", "modal_price", "date"]], use_container_width=True)
                
                # Price trends
                st.markdown("#### Price Trends")
                trends = market_service.get_price_trends(crop_type, 7)
                if trends:
                    trend_df = pd.DataFrame(trends)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df["date"],
                        y=trend_df["price"],
                        mode="lines+markers",
                        name="Price",
                        line=dict(color="green")
                    ))
                    
                    fig.update_layout(
                        title=f"{crop_type.title()} Price Trends (7 days)",
                        xaxis_title="Date",
                        yaxis_title="Price (₹/quintal)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Market recommendations
                recommendations = market_service.get_market_recommendations(prices, crop_type)
                if recommendations:
                    st.markdown(f"#### {get_translation('recommendations', current_lang)}")
                    for rec in recommendations:
                        if rec["priority"] == "high":
                            st.success(f"{rec['icon']} {rec['message']}")
                        elif rec["priority"] == "medium":
                            st.warning(f"{rec['icon']} {rec['message']}")
                        else:
                            st.info(f"{rec['icon']} {rec['message']}")
            else:
                st.info(f"No market data available for {crop_type}. Showing all available crops.")
                
                # Show all prices
                import pandas as pd
                df = pd.DataFrame(prices)
                st.dataframe(df[["commodity", "variety", "market", "modal_price", "date"]], use_container_width=True)
        
        # Market insights
        if prices:
            insights = get_market_insights(prices)
            st.markdown("#### Market Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Price", f"₹{insights['average_price']:.0f}/quintal")
            
            with col2:
                st.metric("Price Range", f"₹{insights['price_range']['min']:.0f} - ₹{insights['price_range']['max']:.0f}")
            
            with col3:
                st.metric("Market Activity", insights['market_activity'])
    
    # Footer
    st.markdown("---")
    footer_text = {
        "en": "🌱 Plant Disease Detection System for Indian Farmers<br>Built with PyTorch and Streamlit | Designed for agricultural communities",
        "hi": "🌱 भारतीय किसानों के लिए पौधे की बीमारी का पता लगाने वाली प्रणाली<br>PyTorch और Streamlit के साथ निर्मित | कृषि समुदायों के लिए डिज़ाइन किया गया",
        "kn": "🌱 ಭಾರತೀಯ ರೈತರಿಗೆ ಸಸ್ಯ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ<br>PyTorch ಮತ್ತು Streamlit ನೊಂದಿಗೆ ನಿರ್ಮಿಸಲಾಗಿದೆ | ಕೃಷಿ ಸಮುದಾಯಗಳಿಗಾಗಿ ವಿನ್ಯಾಸಗೊಳಿಸಲಾಗಿದೆ"
    }
    
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
        <p>{footer_text.get(current_lang, footer_text['en'])}</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
