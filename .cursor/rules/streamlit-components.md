---
type: Auto Attached
globs: ["**/streamlit_app/**", "**/*app.py", "**/*.py"]
description: Streamlit component standards for agriculture UI
---

# Streamlit Component Standards

## App Structure
```python
import streamlit as st
from typing import Dict, List, Optional

# Always define proper type hints for functions
def create_component(data: Dict, config: Optional[Dict] = None) -> None:
    """Component logic here with proper docstrings"""
    # Component implementation
    st.markdown("### Component Title")
    # Streamlit UI elements
    return None
```

## Mobile-First Design Principles
- Use `st.set_page_config(layout="wide")` for responsive design
- Use `st.columns()` for responsive layouts
- Implement touch-friendly UI with `st.button()` and `st.selectbox()`
- Use `st.sidebar` for navigation and secondary features
- Apply custom CSS for mobile optimization with `st.markdown()`

## Accessibility Requirements
- Use descriptive labels with `st.markdown()` for headings
- Implement proper form labels with `st.text_input(label="")`
- Use Streamlit's built-in accessibility features
- Add alt text for images with `st.image(caption="")`
- Use semantic Streamlit components (st.header, st.subheader)

## Multilingual Support
```python
# Use translation functions consistently
def get_text(key: str, **kwargs) -> str:
    """Get translated text for given key"""
    translations = {
        'common.save': {'en': 'Save', 'hi': 'सहेजें'},
        'weather.temperature': {'en': f'Temperature: {kwargs.get("temp", 0)}°C', 
                               'hi': f'तापमान: {kwargs.get("temp", 0)}°C'}
    }
    lang = st.session_state.get('language', 'en')
    return translations.get(key, {}).get(lang, key)

# Example usage
st.button(get_text('common.save'))
st.markdown(get_text('weather.temperature', temp=25))
```

## Agriculture-Specific UI Patterns
```python
# Weather display with icons
def display_weather(temperature: float, humidity: float) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Temperature", f"{temperature}°C", delta="2°C")
    with col2:
        st.metric("Humidity", f"{humidity}%", delta="-5%")

# Soil health with color coding
def display_soil_health(health_score: float) -> None:
    if health_score > 0.7:
        st.success("🟢 Soil Health: Excellent")
    elif health_score > 0.4:
        st.warning("🟡 Soil Health: Moderate")
    else:
        st.error("🔴 Soil Health: Poor")

# Progress indicators for ML processing
def process_disease_detection(image) -> None:
    with st.spinner("Analyzing image for diseases..."):
        # ML processing logic
        result = analyze_image(image)
    st.success("Analysis complete!")
```
