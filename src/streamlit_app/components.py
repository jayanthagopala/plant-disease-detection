"""Custom Streamlit components for plant disease detection."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image


def create_confidence_gauge(confidence: float, title: str = "Confidence Score") -> go.Figure:
    """Create a gauge chart for confidence score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_prediction_radar_chart(predictions: List[Dict[str, float]]) -> go.Figure:
    """Create a radar chart for multiple predictions."""
    categories = [pred['class_name'].replace('_', ' ').title() for pred in predictions]
    values = [pred['confidence'] * 100 for pred in predictions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Confidence',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Prediction Confidence Radar Chart",
        height=400
    )
    
    return fig


def create_disease_comparison_chart(predictions: List[Dict[str, float]]) -> go.Figure:
    """Create a comparison chart for disease predictions."""
    diseases = [pred['class_name'].replace('_', ' ').title() for pred in predictions]
    confidences = [pred['confidence'] * 100 for pred in predictions]
    
    # Color based on confidence level
    colors = []
    for conf in confidences:
        if conf >= 80:
            colors.append('green')
        elif conf >= 60:
            colors.append('orange')
        else:
            colors.append('red')
    
    fig = go.Figure(data=[
        go.Bar(
            x=diseases,
            y=confidences,
            marker_color=colors,
            text=[f"{conf:.1f}%" for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Disease Prediction Comparison",
        xaxis_title="Disease Type",
        yaxis_title="Confidence (%)",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


def create_image_analysis_dashboard(
    image_info: Dict,
    predictions: List[Dict[str, float]],
    disease_info: Dict
) -> None:
    """Create a comprehensive analysis dashboard."""
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Predictions", "📋 Disease Info", "📈 Analytics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image Analysis")
            st.metric("Image Width", f"{image_info.get('width', 'N/A')}px")
            st.metric("Image Height", f"{image_info.get('height', 'N/A')}px")
            st.metric("Image Format", image_info.get('format', 'N/A'))
            st.metric("Color Mode", image_info.get('mode', 'N/A'))
        
        with col2:
            if predictions:
                top_pred = predictions[0]
                st.plotly_chart(
                    create_confidence_gauge(top_pred['confidence'], "Top Prediction Confidence"),
                    use_container_width=True
                )
    
    with tab2:
        st.subheader("Prediction Results")
        
        # Top prediction
        if predictions:
            top_pred = predictions[0]
            st.success(f"**Top Prediction:** {top_pred['class_name'].replace('_', ' ').title()}")
            st.metric("Confidence", f"{top_pred['confidence']:.2%}")
        
        # All predictions chart
        if len(predictions) > 1:
            st.plotly_chart(
                create_disease_comparison_chart(predictions),
                use_container_width=True
            )
        
        # Predictions table
        if predictions:
            df = pd.DataFrame([
                {
                    'Rank': i + 1,
                    'Disease': pred['class_name'].replace('_', ' ').title(),
                    'Confidence': f"{pred['confidence']:.2%}",
                    'Confidence_Value': pred['confidence']
                }
                for i, pred in enumerate(predictions)
            ])
            st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.subheader("Disease Information")
        
        if disease_info:
            st.markdown(f"### {disease_info.get('name', 'Unknown Disease')}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Symptoms")
                st.info(disease_info.get('symptoms', 'Information not available'))
                
                st.markdown("#### Treatment")
                st.warning(disease_info.get('treatment', 'Information not available'))
            
            with col2:
                st.markdown("#### Prevention")
                st.success(disease_info.get('prevention', 'Information not available'))
                
                # Additional info if available
                if 'additional_info' in disease_info:
                    st.markdown("#### Additional Information")
                    st.info(disease_info['additional_info'])
    
    with tab4:
        st.subheader("Analytics")
        
        if predictions:
            # Confidence distribution
            confidences = [pred['confidence'] for pred in predictions]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_prediction_radar_chart(predictions),
                    use_container_width=True
                )
            
            with col2:
                # Confidence statistics
                st.metric("Highest Confidence", f"{max(confidences):.2%}")
                st.metric("Average Confidence", f"{np.mean(confidences):.2%}")
                st.metric("Confidence Range", f"{min(confidences):.2%} - {max(confidences):.2%}")
                
                # Prediction reliability
                high_conf_count = sum(1 for c in confidences if c > 0.8)
                st.metric("High Confidence Predictions", f"{high_conf_count}/{len(confidences)}")


def create_model_performance_widget(
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float
) -> None:
    """Create a widget showing model performance metrics."""
    
    st.subheader("Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col2:
        st.metric("Precision", f"{precision:.2%}")
    
    with col3:
        st.metric("Recall", f"{recall:.2%}")
    
    with col4:
        st.metric("F1-Score", f"{f1_score:.2%}")


def create_upload_guidelines() -> None:
    """Display guidelines for image upload."""
    
    with st.expander("📋 Image Upload Guidelines", expanded=False):
        st.markdown("""
        **For best results, please follow these guidelines when uploading images:**
        
        📸 **Image Quality:**
        - Use high-resolution images (at least 224x224 pixels)
        - Ensure good lighting and clear visibility
        - Avoid blurry or dark images
        
        🍃 **Plant Focus:**
        - Focus on individual leaves or plant parts
        - Include the affected area clearly
        - Avoid images with multiple plants
        
        🌿 **Supported Crops:**
        - Rice, Wheat, Maize, Tomato, Potato
        - Sugarcane, Cotton, and other common Indian crops
        
        📱 **File Formats:**
        - JPG, JPEG, PNG, BMP formats are supported
        - Maximum file size: 10MB
        
        ⚠️ **Important Notes:**
        - The model works best with leaf images
        - Ensure the disease symptoms are visible
        - For multiple diseases, focus on the most prominent one
        """)


def create_footer() -> None:
    """Create application footer."""
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🌱 About This App**
        
        This plant disease detection system is designed specifically for Indian farmers to help identify common crop diseases and get treatment recommendations.
        """)
    
    with col2:
        st.markdown("""
        **🔬 Technology**
        
        Built using PyTorch for deep learning and Streamlit for the user interface. Uses transfer learning with pre-trained CNN models.
        """)
    
    with col3:
        st.markdown("""
        **📞 Support**
        
        For technical support or to report issues, please contact your local agricultural extension office.
        """)
    
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🌱 Plant Disease Detection System | Built for Indian Farmers | Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)


def create_demo_banner() -> None:
    """Create a demo banner for the mock version."""
    st.markdown("""
    <div style="background-color: #e1f5fe; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #0277bd;">
        <h4 style="color: #0277bd; margin: 0;">🚀 Demo Mode</h4>
        <p style="margin: 0.5rem 0 0 0; color: #01579b;">This is a demonstration using mock data. Upload any image to see how the system works!</p>
    </div>
    """, unsafe_allow_html=True)


def create_crop_selector() -> str:
    """Create a crop type selector component."""
    st.markdown("#### 🌾 Select Crop Type")
    
    crops = ["rice", "wheat", "tomato", "potato"]
    selected_crop = st.selectbox(
        "Choose the crop type for better predictions:",
        crops,
        help="Selecting the correct crop type improves prediction accuracy"
    )
    
    return selected_crop


def create_image_preview(image: Image.Image) -> None:
    """Create an enhanced image preview component."""
    st.markdown("#### 📷 Image Preview")
    
    # Resize image for better display
    display_image = image.copy()
    if display_image.width > 400 or display_image.height > 400:
        display_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
    
    st.image(display_image, caption="Uploaded Image", use_column_width=True)
    
    # Image statistics
    st.markdown("##### Image Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Width", f"{image.width}px")
        st.metric("Height", f"{image.height}px")
    
    with col2:
        st.metric("Aspect Ratio", f"{image.width/image.height:.2f}")
        st.metric("Total Pixels", f"{image.width * image.height:,}")
    
    with col3:
        st.metric("Format", image.format or "Unknown")
        st.metric("Mode", image.mode)


def create_prediction_summary(predictions: List[Dict[str, float]]) -> None:
    """Create a summary of prediction results."""
    if not predictions:
        st.warning("No predictions available")
        return
    
    top_pred = predictions[0]
    
    # Confidence level indicator
    confidence = top_pred['confidence']
    if confidence > 0.8:
        confidence_level = "High"
        confidence_color = "green"
        confidence_icon = "🟢"
    elif confidence > 0.6:
        confidence_level = "Medium"
        confidence_color = "orange"
        confidence_icon = "🟡"
    else:
        confidence_level = "Low"
        confidence_color = "red"
        confidence_icon = "🔴"
    
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h4 style="margin: 0 0 0.5rem 0;">{confidence_icon} Top Prediction</h4>
        <h3 style="margin: 0 0 0.5rem 0; color: {confidence_color};">{top_pred['class_name'].replace('_', ' ').title()}</h3>
        <p style="margin: 0;"><strong>Confidence:</strong> <span style="color: {confidence_color};">{confidence:.1%}</span> ({confidence_level})</p>
    </div>
    """, unsafe_allow_html=True)


def create_quick_actions() -> None:
    """Create quick action buttons."""
    st.markdown("#### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View All Predictions", use_container_width=True):
            st.session_state.show_all_predictions = True
    
    with col2:
        if st.button("📋 Disease Info", use_container_width=True):
            st.session_state.show_disease_info = True
    
    with col3:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.clear_analysis = True


def create_mobile_friendly_header() -> None:
    """Create a mobile-friendly header."""
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
        }
        .sub-header {
            font-size: 1.2rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered disease detection for Indian farmers</p>', unsafe_allow_html=True)
