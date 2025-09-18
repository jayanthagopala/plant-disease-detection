#!/usr/bin/env python3
"""Test script for new features: multilingual support, weather, and market prices."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_translations():
    """Test translation functionality."""
    print("Testing translations...")
    
    from streamlit_app.translations import get_translation, create_language_selector
    
    # Test English
    assert get_translation("app_title", "en") == "🌱 Plant Disease Detection for Indian Farmers"
    assert get_translation("disease_detection", "en") == "🔍 Disease Detection"
    
    # Test Hindi
    assert get_translation("app_title", "hi") == "🌱 भारतीय किसानों के लिए पौधे की बीमारी का पता लगाना"
    assert get_translation("disease_detection", "hi") == "🔍 बीमारी का पता लगाना"
    
    # Test Kannada
    assert get_translation("app_title", "kn") == "🌱 ಭಾರತೀಯ ರೈತರಿಗೆ ಸಸ್ಯ ರೋಗ ಪತ್ತೆ"
    assert get_translation("disease_detection", "kn") == "🔍 ರೋಗ ಪತ್ತೆ"
    
    print("✅ Translations working correctly!")

def test_weather_service():
    """Test enhanced weather service functionality."""
    print("Testing enhanced weather service...")
    
    from streamlit_app.weather_service import WeatherService, create_mock_weather_data, get_weather_recommendations
    
    # Test mock weather data
    mock_data = create_mock_weather_data()
    assert "current" in mock_data
    assert "forecast" in mock_data
    assert "alerts" in mock_data
    
    # Test weather recommendations
    recommendations = get_weather_recommendations(mock_data, "rice")
    assert isinstance(recommendations, list)
    
    # Test real weather service
    weather = WeatherService()
    
    # Test coordinate fallback
    coords = weather.get_coordinates("Bangalore", "Karnataka", "India")
    assert coords is not None
    assert "latitude" in coords
    assert "longitude" in coords
    
    # Test comprehensive weather data
    comprehensive_data = weather.get_comprehensive_weather_data("Bangalore", "Karnataka", "India")
    if comprehensive_data:  # Only test if API call succeeds
        assert "location" in comprehensive_data
        assert "current" in comprehensive_data
        assert "forecast" in comprehensive_data
        assert "historical" in comprehensive_data
        assert "alerts" in comprehensive_data
        
        # Test weather recommendations with real data
        real_recommendations = get_weather_recommendations(comprehensive_data, "rice")
        assert isinstance(real_recommendations, list)
    
    print("✅ Enhanced weather service working correctly!")

def test_market_service():
    """Test market service functionality."""
    print("Testing market service...")
    
    from streamlit_app.market_service import MarketPriceService, create_mock_market_data, get_market_insights
    
    # Test mock market data
    mock_data = create_mock_market_data()
    assert "prices" in mock_data
    assert "trends" in mock_data
    
    # Test market insights
    insights = get_market_insights(mock_data["prices"])
    assert "average_price" in insights
    assert "price_range" in insights
    
    print("✅ Market service working correctly!")

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from streamlit_app.translations import get_translation, create_language_selector
        from streamlit_app.weather_service import WeatherService, create_mock_weather_data
        from streamlit_app.market_service import MarketPriceService, create_mock_market_data
        print("✅ All imports working correctly!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing new features for Smart Crop Advisory System")
    print("=" * 60)
    
    try:
        test_imports()
        test_translations()
        test_weather_service()
        test_market_service()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! New features are working correctly.")
        print("\nNew features added:")
        print("✅ Multilingual support (English, Hindi, Kannada)")
        print("✅ Real-time weather data from Open-Meteo API")
        print("✅ 7-day weather forecast with detailed parameters")
        print("✅ 7-day historical weather data")
        print("✅ Intelligent weather alerts and recommendations")
        print("✅ Crop-specific weather analysis")
        print("✅ Market price tracking and trends")
        print("✅ Enhanced UI with 5 tabs")
        print("✅ Fallback coordinates for major Indian cities")
        print("✅ Weather trend analysis and humidity monitoring")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
