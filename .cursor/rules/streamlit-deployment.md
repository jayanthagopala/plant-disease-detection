---
type: Auto Attached
globs: ["**/streamlit_app/**", "**/*app.py", "**/requirements.txt", "**/Dockerfile"]
description: Streamlit deployment configuration for rural connectivity
---

# Streamlit Deployment Guidelines

## Deployment Requirements for Rural Areas
- Deploy on reliable cloud platforms (Streamlit Cloud, Heroku, AWS)
- Implement caching for API responses and model predictions
- Optimize for slow network connections
- Use mobile-responsive Streamlit configurations

## Caching Strategy
```python
import streamlit as st
from datetime import datetime, timedelta
import pickle
import os

# Cache API responses in Streamlit session state
@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_weather_data(latitude: float, longitude: float) -> dict:
    """Fetch weather data with caching"""
    try:
        # API call logic here
        response = fetch_weather_api(latitude, longitude)
        return response
    except Exception as e:
        st.error(f"Failed to fetch weather data: {e}")
        return get_fallback_weather_data()

# Cache model predictions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_disease_prediction(image_bytes: bytes) -> dict:
    """Cache disease prediction results"""
    # Model prediction logic here
    return prediction_result
```

## Streamlit Configuration
```python
# config.toml for Streamlit Cloud deployment
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[browser]
gatherUsageStats = false

# Mobile optimization
[client]
showErrorDetails = true
```

## Performance Optimization
```python
# Optimize for slow connections
import streamlit as st

# Use session state for data persistence
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# Implement lazy loading for images
def load_image_optimized(image_path: str):
    """Load images with optimization for slow connections"""
    try:
        return st.image(image_path, width=300, caption="Crop Image")
    except Exception as e:
        st.error("Failed to load image. Please check your connection.")

# Add loading indicators
with st.spinner("Loading crop data..."):
    # Heavy operations here
    data = load_crop_data()

# Use progress bars for long operations
progress_bar = st.progress(0)
for i in range(100):
    # Simulate work
    progress_bar.progress(i + 1)
```
