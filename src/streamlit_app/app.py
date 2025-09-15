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

from models.disease_classifier import DiseaseClassifier
from data.utils import create_sample_dataset


# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .disease-info {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffa500;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str, device: str = "cpu") -> DiseaseClassifier:
    """Load the trained model with caching."""
    try:
        classifier = DiseaseClassifier(
            model_path=Path(model_path),
            device=device
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_sample_data_if_needed(data_dir: Path) -> None:
    """Create sample data if the data directory is empty."""
    if not data_dir.exists() or not any(data_dir.iterdir()):
        with st.spinner("Creating sample dataset..."):
            create_sample_dataset(data_dir, num_samples_per_class=20)
        st.success("Sample dataset created!")


def display_prediction_results(
    predictions: List[Dict[str, float]], 
    classifier: DiseaseClassifier
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
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_path = st.sidebar.selectbox(
        "Select Model",
        ["outputs/best_model.pth", "outputs/final_model.pth"],
        help="Choose the trained model to use for predictions"
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
    
    # Load model
    if st.sidebar.button("🔄 Load Model"):
        with st.spinner("Loading model..."):
            classifier = load_model(model_path, device)
            if classifier:
                st.sidebar.success("Model loaded successfully!")
            else:
                st.sidebar.error("Failed to load model!")
    
    # Check if model is loaded
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    
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
        sample_dir = Path("data/plant_diseases")
        create_sample_data_if_needed(sample_dir)
        
        if sample_dir.exists():
            sample_classes = [d.name for d in sample_dir.iterdir() if d.is_dir()]
            if sample_classes:
                selected_class = st.selectbox("Select sample class:", sample_classes)
                sample_images = list((sample_dir / selected_class).glob("*.jpg"))
                
                if sample_images:
                    selected_sample = st.selectbox(
                        "Select sample image:", 
                        [img.name for img in sample_images]
                    )
                    
                    if st.button("Use Sample Image"):
                        sample_path = sample_dir / selected_class / selected_sample
                        uploaded_file = sample_path
    
    with col2:
        st.markdown("### 🔍 Analysis")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown("#### Image Information")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Width", f"{image.width}px")
                st.metric("Height", f"{image.height}px")
            with col_info2:
                st.metric("Format", image.format)
                st.metric("Mode", image.mode)
            
            # Predict button
            if st.button("🔍 Analyze Image", type="primary"):
                if st.session_state.classifier is None:
                    st.error("Please load a model first!")
                else:
                    with st.spinner("Analyzing image..."):
                        try:
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
