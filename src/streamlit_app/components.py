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


def create_weather_forecast_chart(forecast_data: List[Dict], current_weather: Dict = None, language: str = "en") -> go.Figure:
    """Create an enhanced weather forecast chart."""
    if not forecast_data:
        return None
    
    # Import translations
    from translations import get_translation
    
    # Prepare data
    dates = [day["date"] for day in forecast_data]
    max_temps = [day["max_temp"] for day in forecast_data]
    min_temps = [day["min_temp"] for day in forecast_data]
    precipitation = [day["precipitation"] for day in forecast_data]
    precipitation_prob = [day.get("precipitation_prob", 0) for day in forecast_data]
    weather_codes = [day.get("weather_code", 0) for day in forecast_data]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            get_translation("temperature_forecast", language), 
            get_translation("precipitation_forecast", language), 
            get_translation("weather_conditions", language)
        ),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Temperature chart
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=max_temps, 
            name="Max Temperature", 
            line=dict(color="red", width=3),
            marker=dict(size=8, color="red")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=min_temps, 
            name="Min Temperature", 
            line=dict(color="blue", width=3),
            marker=dict(size=8, color="blue"),
            fill="tonexty"
        ),
        row=1, col=1
    )
    
    # Precipitation chart
    fig.add_trace(
        go.Bar(
            x=dates, 
            y=precipitation, 
            name="Precipitation (mm)",
            marker_color="lightblue",
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Precipitation probability
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=precipitation_prob,
            name="Rain Probability (%)",
            line=dict(color="darkblue", width=2, dash="dash"),
            marker=dict(size=6, color="darkblue")
        ),
        row=2, col=1
    )
    
    # Weather conditions (using weather codes as bar chart)
    weather_descriptions = [day.get("description", "Unknown") for day in forecast_data]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=weather_codes,
            name="Weather Code",
            marker_color="lightgreen",
            opacity=0.6,
            text=weather_descriptions,
            textposition="auto"
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title="7-Day Weather Forecast",
        title_x=0.5
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Precipitation (mm)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Weather Code", row=3, col=1)
    
    return fig


def create_weather_alerts_display(alerts: List[Dict]) -> None:
    """Create an enhanced weather alerts display."""
    if not alerts:
        st.info("🌤️ No weather alerts at this time. Conditions are normal.")
        return
    
    st.markdown("#### 🚨 Weather Alerts")
    
    # Group alerts by priority
    high_priority = [alert for alert in alerts if alert.get("priority") == "high"]
    medium_priority = [alert for alert in alerts if alert.get("priority") == "medium"]
    low_priority = [alert for alert in alerts if alert.get("priority") == "low"]
    
    # Display high priority alerts
    if high_priority:
        for alert in high_priority:
            st.markdown(f"""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                <h4 style="color: #721c24; margin: 0 0 0.5rem 0;">{alert.get('icon', '⚠️')} {alert.get('type', 'Alert').title()}</h4>
                <p style="color: #721c24; margin: 0;">{alert.get('message', 'No message available')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display medium priority alerts
    if medium_priority:
        for alert in medium_priority:
            st.markdown(f"""
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                <h4 style="color: #856404; margin: 0 0 0.5rem 0;">{alert.get('icon', '⚠️')} {alert.get('type', 'Alert').title()}</h4>
                <p style="color: #856404; margin: 0;">{alert.get('message', 'No message available')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display low priority alerts
    if low_priority:
        for alert in low_priority:
            st.markdown(f"""
            <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                <h4 style="color: #0c5460; margin: 0 0 0.5rem 0;">{alert.get('icon', 'ℹ️')} {alert.get('type', 'Info').title()}</h4>
                <p style="color: #0c5460; margin: 0;">{alert.get('message', 'No message available')}</p>
            </div>
            """, unsafe_allow_html=True)


def create_weather_recommendations_display(recommendations: List[Dict]) -> None:
    """Create an enhanced weather recommendations display."""
    if not recommendations:
        st.info("🌱 No specific weather recommendations at this time.")
        return
    
    st.markdown("#### 🌾 Weather-Based Recommendations")
    
    # Group recommendations by type
    irrigation_recs = [rec for rec in recommendations if rec.get("type") == "irrigation"]
    protection_recs = [rec for rec in recommendations if rec.get("type") == "protection"]
    disease_recs = [rec for rec in recommendations if rec.get("type") == "disease_prevention"]
    drainage_recs = [rec for rec in recommendations if rec.get("type") == "drainage"]
    crop_specific_recs = [rec for rec in recommendations if rec.get("type") == "crop_specific"]
    other_recs = [rec for rec in recommendations if rec.get("type") not in ["irrigation", "protection", "disease_prevention", "drainage", "crop_specific"]]
    
    # Display irrigation recommendations
    if irrigation_recs:
        with st.expander("💧 Irrigation Recommendations", expanded=True):
            for rec in irrigation_recs:
                priority_color = "red" if rec.get("priority") == "critical" else "orange" if rec.get("priority") == "high" else "blue"
                st.markdown(f"""
                <div style="background-color: #e3f2fd; border-left: 4px solid {priority_color}; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: {priority_color};">{rec.get('icon', '💧')} {rec.get('message', 'No message')}</h4>
                    <p style="margin: 0; font-weight: bold;">Action: {rec.get('action', 'No action specified')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display protection recommendations
    if protection_recs:
        with st.expander("🛡️ Crop Protection Recommendations", expanded=True):
            for rec in protection_recs:
                priority_color = "red" if rec.get("priority") == "critical" else "orange" if rec.get("priority") == "high" else "blue"
                st.markdown(f"""
                <div style="background-color: #fff3e0; border-left: 4px solid {priority_color}; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: {priority_color};">{rec.get('icon', '🛡️')} {rec.get('message', 'No message')}</h4>
                    <p style="margin: 0; font-weight: bold;">Action: {rec.get('action', 'No action specified')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display disease prevention recommendations
    if disease_recs:
        with st.expander("🦠 Disease Prevention Recommendations", expanded=True):
            for rec in disease_recs:
                priority_color = "red" if rec.get("priority") == "critical" else "orange" if rec.get("priority") == "high" else "blue"
                st.markdown(f"""
                <div style="background-color: #fce4ec; border-left: 4px solid {priority_color}; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: {priority_color};">{rec.get('icon', '🦠')} {rec.get('message', 'No message')}</h4>
                    <p style="margin: 0; font-weight: bold;">Action: {rec.get('action', 'No action specified')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display drainage recommendations
    if drainage_recs:
        with st.expander("🌊 Drainage Recommendations", expanded=True):
            for rec in drainage_recs:
                priority_color = "red" if rec.get("priority") == "critical" else "orange" if rec.get("priority") == "high" else "blue"
                st.markdown(f"""
                <div style="background-color: #e0f2f1; border-left: 4px solid {priority_color}; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: {priority_color};">{rec.get('icon', '🌊')} {rec.get('message', 'No message')}</h4>
                    <p style="margin: 0; font-weight: bold;">Action: {rec.get('action', 'No action specified')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display crop-specific recommendations
    if crop_specific_recs:
        with st.expander("🌾 Crop-Specific Recommendations", expanded=True):
            for rec in crop_specific_recs:
                priority_color = "red" if rec.get("priority") == "critical" else "orange" if rec.get("priority") == "high" else "blue"
                st.markdown(f"""
                <div style="background-color: #f1f8e9; border-left: 4px solid {priority_color}; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: {priority_color};">{rec.get('icon', '🌾')} {rec.get('message', 'No message')}</h4>
                    <p style="margin: 0; font-weight: bold;">Action: {rec.get('action', 'No action specified')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display other recommendations
    if other_recs:
        with st.expander("📋 Other Recommendations", expanded=False):
            for rec in other_recs:
                priority_color = "red" if rec.get("priority") == "critical" else "orange" if rec.get("priority") == "high" else "blue"
                st.markdown(f"""
                <div style="background-color: #f5f5f5; border-left: 4px solid {priority_color}; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: {priority_color};">{rec.get('icon', '📋')} {rec.get('message', 'No message')}</h4>
                    <p style="margin: 0; font-weight: bold;">Action: {rec.get('action', 'No action specified')}</p>
                </div>
                """, unsafe_allow_html=True)


def create_current_weather_display(current_weather: Dict) -> None:
    """Create an enhanced current weather display."""
    if not current_weather:
        st.warning("Unable to fetch current weather data.")
        return
    
    st.markdown("#### 🌤️ Current Weather Conditions")
    
    # Main weather metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = current_weather.get("temperature", 0)
        temp_color = "red" if temp > 35 else "blue" if temp < 15 else "green"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; border: 2px solid {temp_color};">
            <h2 style="margin: 0; color: {temp_color};">{temp:.1f}°C</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Temperature</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        humidity = current_weather.get("humidity", 0)
        humidity_color = "red" if humidity > 80 else "orange" if humidity > 60 else "green"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; border: 2px solid {humidity_color};">
            <h2 style="margin: 0; color: {humidity_color};">{humidity:.0f}%</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Humidity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        precipitation = current_weather.get("precipitation", 0)
        precip_color = "red" if precipitation > 10 else "orange" if precipitation > 5 else "green"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; border: 2px solid {precip_color};">
            <h2 style="margin: 0; color: {precip_color};">{precipitation:.1f}mm</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Rainfall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        wind_speed = current_weather.get("wind_speed", 0)
        wind_color = "red" if wind_speed > 25 else "orange" if wind_speed > 15 else "green"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; border: 2px solid {wind_color};">
            <h2 style="margin: 0; color: {wind_color};">{wind_speed:.1f} km/h</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Wind Speed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional weather details
    st.markdown("#### 📊 Additional Weather Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Weather Description", current_weather.get("description", "Unknown"))
        st.metric("Cloud Cover", f"{current_weather.get('cloud_cover', 0):.0f}%")
    
    with col2:
        st.metric("Wind Direction", f"{current_weather.get('wind_direction', 0):.0f}°")
        st.metric("Atmospheric Pressure", f"{current_weather.get('pressure', 0):.1f} hPa")
    
    with col3:
        st.metric("Last Updated", current_weather.get("timestamp", "Unknown")[:16])


def create_weather_summary_card(weather_data: Dict) -> None:
    """Create a weather summary card for quick overview."""
    if not weather_data:
        return
    
    current = weather_data.get("current", {})
    forecast = weather_data.get("forecast", [])
    alerts = weather_data.get("alerts", [])
    
    # Calculate summary metrics
    temp = current.get("temperature", 0)
    humidity = current.get("humidity", 0)
    precipitation = current.get("precipitation", 0)
    
    # Weather condition assessment
    if temp > 35:
        condition = "Hot"
        condition_icon = "🔥"
        condition_color = "red"
    elif temp < 15:
        condition = "Cold"
        condition_icon = "❄️"
        condition_color = "blue"
    else:
        condition = "Moderate"
        condition_icon = "🌤️"
        condition_color = "green"
    
    # Alert count
    alert_count = len(alerts)
    alert_text = f"{alert_count} alert{'s' if alert_count != 1 else ''}"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0 0 0.5rem 0;">{condition_icon} {condition} Weather</h3>
                <p style="margin: 0; font-size: 1.1rem;">{temp:.1f}°C • {humidity:.0f}% humidity • {precipitation:.1f}mm rain</p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 0.9rem;">{alert_text}</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">7-day forecast available</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_city_selector(weather_service, current_lang: str = "en") -> tuple:
    """Create a city selector component for Karnataka cities."""
    from translations import get_translation
    
    st.markdown("#### 🌍 Select Your City")
    
    # Get Karnataka cities
    karnataka_cities = weather_service.get_karnataka_cities()
    
    # Create city options
    city_options = {city["name"]: city["key"] for city in karnataka_cities}
    
    # Add search functionality
    search_term = st.text_input(
        "🔍 Search for your city:",
        placeholder="Type to search cities...",
        help="Start typing to filter cities"
    )
    
    # Filter cities based on search
    if search_term:
        filtered_cities = {
            name: key for name, key in city_options.items() 
            if search_term.lower() in name.lower()
        }
    else:
        filtered_cities = city_options
    
    # Display city selector
    if filtered_cities:
        selected_city_display = st.selectbox(
            get_translation("select_city", current_lang),
            list(filtered_cities.keys()),
            help="Select your city for accurate weather data"
        )
        selected_city_key = filtered_cities[selected_city_display]
    else:
        st.warning("No cities found matching your search. Please try a different search term.")
        selected_city_display = "Bangalore"
        selected_city_key = "bangalore"
    
    # Display selected city info
    if selected_city_key in weather_service.fallback_coordinates:
        coords = weather_service.fallback_coordinates[selected_city_key]
        st.info(f"📍 Selected: {selected_city_display}, {coords.get('state', 'Karnataka')}")
    
    return selected_city_key, selected_city_display


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
