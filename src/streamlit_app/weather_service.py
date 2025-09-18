"""Weather service for the Smart Crop Advisory System."""

import requests
import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json

class WeatherService:
    """Weather service using Open-Meteo API (free tier)."""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    
    def get_coordinates(self, city: str, state: str = "Karnataka", country: str = "India") -> Optional[Dict[str, float]]:
        """Get coordinates for a city."""
        try:
            params = {
                "name": f"{city}, {state}, {country}",
                "count": 1,
                "language": "en",
                "format": "json"
            }
            
            response = requests.get(self.geocoding_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                return {
                    "latitude": result["latitude"],
                    "longitude": result["longitude"]
                }
        except Exception as e:
            st.warning(f"Could not get coordinates for {city}: {e}")
        
        return None
    
    def get_current_weather(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """Get current weather data."""
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "timezone": "auto"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data.get("current", {})
            
            return {
                "temperature": current.get("temperature_2m", 0),
                "humidity": current.get("relative_humidity_2m", 0),
                "precipitation": current.get("precipitation", 0),
                "wind_speed": current.get("wind_speed_10m", 0),
                "timestamp": current.get("time", datetime.now().isoformat())
            }
        except Exception as e:
            st.warning(f"Could not fetch current weather: {e}")
            return None
    
    def get_weather_forecast(self, latitude: float, longitude: float, days: int = 7) -> Optional[Dict[str, Any]]:
        """Get weather forecast for the next few days."""
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": days
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            daily = data.get("daily", {})
            
            forecast = []
            for i in range(len(daily.get("time", []))):
                forecast.append({
                    "date": daily["time"][i],
                    "max_temp": daily["temperature_2m_max"][i],
                    "min_temp": daily["temperature_2m_min"][i],
                    "precipitation": daily["precipitation_sum"][i],
                    "precipitation_prob": daily["precipitation_probability_max"][i]
                })
            
            return forecast
        except Exception as e:
            st.warning(f"Could not fetch weather forecast: {e}")
            return None
    
    def get_weather_alerts(self, latitude: float, longitude: float) -> list:
        """Get weather alerts (mock implementation for demo)."""
        # In a real implementation, this would fetch from weather alert APIs
        alerts = []
        
        # Mock alerts based on weather conditions
        current_weather = self.get_current_weather(latitude, longitude)
        if current_weather:
            if current_weather["precipitation"] > 10:
                alerts.append({
                    "type": "rain",
                    "severity": "moderate",
                    "message": "Heavy rainfall expected. Consider irrigation adjustments.",
                    "icon": "🌧️"
                })
            
            if current_weather["temperature"] > 35:
                alerts.append({
                    "type": "heat",
                    "severity": "high",
                    "message": "High temperature alert. Ensure proper irrigation.",
                    "icon": "🌡️"
                })
            
            if current_weather["wind_speed"] > 15:
                alerts.append({
                    "type": "wind",
                    "severity": "moderate",
                    "message": "Strong winds expected. Protect young plants.",
                    "icon": "💨"
                })
        
        return alerts

def create_mock_weather_data() -> Dict[str, Any]:
    """Create mock weather data for demonstration."""
    return {
        "current": {
            "temperature": 28.5,
            "humidity": 65,
            "precipitation": 2.5,
            "wind_speed": 8.2,
            "timestamp": datetime.now().isoformat()
        },
        "forecast": [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "max_temp": 30 + i,
                "min_temp": 22 - i,
                "precipitation": max(0, 5 - i),
                "precipitation_prob": max(0, 80 - i * 10)
            }
            for i in range(7)
        ],
        "alerts": [
            {
                "type": "rain",
                "severity": "moderate",
                "message": "Moderate rainfall expected tomorrow. Good for irrigation.",
                "icon": "🌧️"
            }
        ]
    }

def get_weather_recommendations(weather_data: Dict[str, Any], crop_type: str) -> list:
    """Get weather-based recommendations for farming."""
    recommendations = []
    
    if not weather_data:
        return recommendations
    
    current = weather_data.get("current", {})
    forecast = weather_data.get("forecast", [])
    
    # Temperature recommendations
    temp = current.get("temperature", 0)
    if temp > 35:
        recommendations.append({
            "type": "irrigation",
            "priority": "high",
            "message": f"High temperature ({temp}°C) detected. Increase irrigation frequency.",
            "icon": "💧"
        })
    elif temp < 15:
        recommendations.append({
            "type": "protection",
            "priority": "medium",
            "message": f"Low temperature ({temp}°C) detected. Protect sensitive crops.",
            "icon": "🛡️"
        })
    
    # Precipitation recommendations
    precipitation = current.get("precipitation", 0)
    if precipitation > 10:
        recommendations.append({
            "type": "drainage",
            "priority": "high",
            "message": f"Heavy rainfall ({precipitation}mm). Ensure proper drainage.",
            "icon": "🌊"
        })
    elif precipitation < 1 and any(day["precipitation"] < 1 for day in forecast[:3]):
        recommendations.append({
            "type": "irrigation",
            "priority": "medium",
            "message": "Dry conditions expected. Plan irrigation schedule.",
            "icon": "💧"
        })
    
    # Wind recommendations
    wind_speed = current.get("wind_speed", 0)
    if wind_speed > 15:
        recommendations.append({
            "type": "protection",
            "priority": "medium",
            "message": f"Strong winds ({wind_speed} km/h). Secure young plants.",
            "icon": "💨"
        })
    
    # Crop-specific recommendations
    if crop_type.lower() == "rice":
        if precipitation > 5:
            recommendations.append({
                "type": "crop_specific",
                "priority": "low",
                "message": "Good rainfall for rice cultivation. Monitor water levels.",
                "icon": "🌾"
            })
    elif crop_type.lower() == "wheat":
        if temp > 30:
            recommendations.append({
                "type": "crop_specific",
                "priority": "medium",
                "message": "High temperature may affect wheat grain filling. Monitor closely.",
                "icon": "🌾"
            })
    
    return recommendations
