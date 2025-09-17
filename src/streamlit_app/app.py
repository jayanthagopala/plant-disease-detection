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

# Import mock data instead of actual model
from mock_data import (
    MockDiseaseClassifier, 
    get_mock_disease_info, 
    create_mock_image_info,
    get_mock_model_performance,
    get_supported_crops,
    create_mock_sample_images,
    DISEASE_CATEGORIES
)


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
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detection for Indian Farmers</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image of your plant to detect diseases and get treatment recommendations</p>', unsafe_allow_html=True)
    
    # Demo notice
    st.info("🚀 **Demo Mode**: This is a demonstration using mock data. Upload any image to see how the system works!")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_path = st.sidebar.selectbox(
        "Select Model",
        ["Mock Model (ResNet-50)", "Mock Model (EfficientNet)", "Mock Model (Custom CNN)"],
        help="Choose the mock model to use for predictions"
    )
    
    # Device selection
    device = st.sidebar.selectbox(
        "Device",
        ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        help="Select the device for inference"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for predictions"
    )
    
    # Crop type selection
    crop_type = st.sidebar.selectbox(
        "Crop Type",
        get_supported_crops(),
        help="Select the crop type for better predictions"
    )
    
    # Load model
    if st.sidebar.button("🔄 Load Model"):
        with st.spinner("Loading model..."):
            classifier = load_model(model_path, device)
            if classifier:
                st.sidebar.success("Mock model loaded successfully!")
                st.session_state.classifier = classifier
            else:
                st.sidebar.error("Failed to load model!")
    
    # Auto-load mock model if not already loaded
    if 'classifier' not in st.session_state:
        st.session_state.classifier = MockDiseaseClassifier()
        st.sidebar.success("Mock model ready!")
    
    # Model performance metrics
    if st.sidebar.checkbox("Show Model Performance", value=True):
        st.sidebar.markdown("#### 📊 Model Performance")
        performance = get_mock_model_performance()
        st.sidebar.metric("Accuracy", f"{performance['accuracy']:.1%}")
        st.sidebar.metric("Precision", f"{performance['precision']:.1%}")
        st.sidebar.metric("Recall", f"{performance['recall']:.1%}")
        st.sidebar.metric("F1-Score", f"{performance['f1_score']:.1%}")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🔍 Disease Detection", "📊 Model Info", "📚 Disease Database"])
    
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
                    image = uploaded_file
                    
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
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
                                # Generate mock predictions based on selected crop type
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
            else:
                st.info("Please upload an image or select a sample image to get started.")
    
    with tab2:
        st.markdown("### 🤖 Model Information")
        
        if st.session_state.classifier:
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
        st.markdown("### 📚 Disease Database")
        
        # Show diseases for selected crop
        if crop_type in DISEASE_CATEGORIES:
            st.markdown(f"#### {crop_type.title()} Diseases")
            
            for disease, info in DISEASE_CATEGORIES[crop_type].items():
                with st.expander(f"🌿 {info['name']}", expanded=False):
                    st.markdown(f"**Symptoms:** {info['symptoms']}")
                    st.markdown(f"**Treatment:** {info['treatment']}")
                    st.markdown(f"**Prevention:** {info['prevention']}")
        else:
            st.info("Select a crop type from the sidebar to view disease information.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌱 Plant Disease Detection System for Indian Farmers</p>
        <p>Built with PyTorch and Streamlit | Designed for agricultural communities</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
