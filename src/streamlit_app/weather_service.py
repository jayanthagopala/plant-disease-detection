"""Weather service for the Smart Crop Advisory System."""

import requests
import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherService:
    """Enhanced weather service using Open-Meteo API (free tier)."""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.historical_url = "https://archive-api.open-meteo.com/v1/archive"
        self.alerts_url = "https://api.open-meteo.com/v1/alert"
        
        # Fallback coordinates for major Indian cities
        self.fallback_coordinates = {
            # Karnataka cities
            "bangalore": {"latitude": 12.9716, "longitude": 77.5946, "state": "Karnataka"},
            "mysore": {"latitude": 12.2958, "longitude": 76.6394, "state": "Karnataka"},
            "hubli": {"latitude": 15.3647, "longitude": 75.1240, "state": "Karnataka"},
            "mangalore": {"latitude": 12.9141, "longitude": 74.8560, "state": "Karnataka"},
            "belgaum": {"latitude": 15.8497, "longitude": 74.4977, "state": "Karnataka"},
            "gulbarga": {"latitude": 17.3297, "longitude": 76.8343, "state": "Karnataka"},
            "davangere": {"latitude": 14.4644, "longitude": 75.9218, "state": "Karnataka"},
            "bellary": {"latitude": 15.1420, "longitude": 76.9210, "state": "Karnataka"},
            "bijapur": {"latitude": 16.8244, "longitude": 75.7154, "state": "Karnataka"},
            "shimoga": {"latitude": 13.9299, "longitude": 75.5681, "state": "Karnataka"},
            "tumkur": {"latitude": 13.3392, "longitude": 77.1139, "state": "Karnataka"},
            "raichur": {"latitude": 16.2076, "longitude": 77.3463, "state": "Karnataka"},
            "bidar": {"latitude": 17.9134, "longitude": 77.5301, "state": "Karnataka"},
            "hospet": {"latitude": 15.2667, "longitude": 76.4000, "state": "Karnataka"},
            "gadag": {"latitude": 15.4319, "longitude": 75.6319, "state": "Karnataka"},
            "chitradurga": {"latitude": 14.2254, "longitude": 76.3985, "state": "Karnataka"},
            "kolar": {"latitude": 13.1372, "longitude": 78.1300, "state": "Karnataka"},
            "mandya": {"latitude": 12.5242, "longitude": 76.8958, "state": "Karnataka"},
            "hassan": {"latitude": 13.0049, "longitude": 76.1025, "state": "Karnataka"},
            "udupi": {"latitude": 13.3409, "longitude": 74.7421, "state": "Karnataka"},
            "chikmagalur": {"latitude": 13.3161, "longitude": 75.7720, "state": "Karnataka"},
            "karwar": {"latitude": 14.8136, "longitude": 74.1297, "state": "Karnataka"},
            "chamrajanagar": {"latitude": 11.9261, "longitude": 76.9397, "state": "Karnataka"},
            "bagalkot": {"latitude": 16.1862, "longitude": 75.6964, "state": "Karnataka"},
            "kolar": {"latitude": 13.1372, "longitude": 78.1300, "state": "Karnataka"},
            "kolar gold fields": {"latitude": 12.9667, "longitude": 78.2667, "state": "Karnataka"},
            "chikballapur": {"latitude": 13.4352, "longitude": 77.7275, "state": "Karnataka"},
            "ramanagara": {"latitude": 12.7234, "longitude": 77.3000, "state": "Karnataka"},
            "chintamani": {"latitude": 13.4000, "longitude": 78.0667, "state": "Karnataka"},
            "gokak": {"latitude": 16.1667, "longitude": 74.8333, "state": "Karnataka"},
            "yadgir": {"latitude": 16.7667, "longitude": 77.1333, "state": "Karnataka"},
            "koppal": {"latitude": 15.3500, "longitude": 76.1500, "state": "Karnataka"},
            "haveri": {"latitude": 14.8000, "longitude": 75.4000, "state": "Karnataka"},
            "ranebennur": {"latitude": 14.6167, "longitude": 75.6167, "state": "Karnataka"},
            "sirsi": {"latitude": 14.6167, "longitude": 74.8333, "state": "Karnataka"},
            "ankola": {"latitude": 14.6667, "longitude": 74.3000, "state": "Karnataka"},
            "kumta": {"latitude": 14.4167, "longitude": 74.4167, "state": "Karnataka"},
            "honnavar": {"latitude": 14.2833, "longitude": 74.4500, "state": "Karnataka"},
            "bhatkal": {"latitude": 13.9667, "longitude": 74.5667, "state": "Karnataka"},
            "murdeshwar": {"latitude": 14.1000, "longitude": 74.4833, "state": "Karnataka"},
            "kundapur": {"latitude": 13.6167, "longitude": 74.6833, "state": "Karnataka"},
            "udupi": {"latitude": 13.3409, "longitude": 74.7421, "state": "Karnataka"},
            "karkala": {"latitude": 13.2000, "longitude": 74.9833, "state": "Karnataka"},
            "sringeri": {"latitude": 13.4167, "longitude": 75.2500, "state": "Karnataka"},
            "chikmagalur": {"latitude": 13.3161, "longitude": 75.7720, "state": "Karnataka"},
            "tarikere": {"latitude": 13.7167, "longitude": 75.8167, "state": "Karnataka"},
            "kadur": {"latitude": 13.5500, "longitude": 76.0167, "state": "Karnataka"},
            "arsikere": {"latitude": 13.3167, "longitude": 76.2500, "state": "Karnataka"},
            "hassan": {"latitude": 13.0049, "longitude": 76.1025, "state": "Karnataka"},
            "holenarasipura": {"latitude": 12.7833, "longitude": 76.2500, "state": "Karnataka"},
            "channarayapatna": {"latitude": 12.9000, "longitude": 76.4000, "state": "Karnataka"},
            "mandya": {"latitude": 12.5242, "longitude": 76.8958, "state": "Karnataka"},
            "maddur": {"latitude": 12.5833, "longitude": 77.0500, "state": "Karnataka"},
            "malavalli": {"latitude": 12.3833, "longitude": 77.0833, "state": "Karnataka"},
            "pandavapura": {"latitude": 12.5000, "longitude": 76.6667, "state": "Karnataka"},
            "srirangapatna": {"latitude": 12.4167, "longitude": 76.7000, "state": "Karnataka"},
            "mysore": {"latitude": 12.2958, "longitude": 76.6394, "state": "Karnataka"},
            "nanjangud": {"latitude": 12.1167, "longitude": 76.6833, "state": "Karnataka"},
            "tirumakudalu narasipura": {"latitude": 12.2167, "longitude": 76.9000, "state": "Karnataka"},
            "chamrajanagar": {"latitude": 11.9261, "longitude": 76.9397, "state": "Karnataka"},
            "kollegal": {"latitude": 12.1500, "longitude": 77.1167, "state": "Karnataka"},
            "yelandur": {"latitude": 12.0500, "longitude": 77.0333, "state": "Karnataka"},
            "gundlupet": {"latitude": 11.8000, "longitude": 76.7000, "state": "Karnataka"},
            "hunsur": {"latitude": 12.3000, "longitude": 76.2833, "state": "Karnataka"},
            "periyapatna": {"latitude": 12.3500, "longitude": 76.2000, "state": "Karnataka"},
            "kushalnagar": {"latitude": 12.4500, "longitude": 75.9500, "state": "Karnataka"},
            "somnathpur": {"latitude": 12.2667, "longitude": 76.8500, "state": "Karnataka"},
            "bannur": {"latitude": 12.3333, "longitude": 76.8667, "state": "Karnataka"},
            "krishnarajanagar": {"latitude": 12.4333, "longitude": 76.3833, "state": "Karnataka"},
            "kabini": {"latitude": 11.9000, "longitude": 76.3500, "state": "Karnataka"},
            "heggadadevanakote": {"latitude": 11.9000, "longitude": 76.3500, "state": "Karnataka"},
            "nagamangala": {"latitude": 12.8167, "longitude": 76.7500, "state": "Karnataka"},
            "pandavapura": {"latitude": 12.5000, "longitude": 76.6667, "state": "Karnataka"},
            "krishnarajpet": {"latitude": 12.6667, "longitude": 76.5000, "state": "Karnataka"},
            "shravanabelagola": {"latitude": 12.8500, "longitude": 76.4833, "state": "Karnataka"},
            "channapatna": {"latitude": 12.6500, "longitude": 77.2167, "state": "Karnataka"},
            "ramanagara": {"latitude": 12.7234, "longitude": 77.3000, "state": "Karnataka"},
            "magadi": {"latitude": 12.9667, "longitude": 77.2333, "state": "Karnataka"},
            "nelamangala": {"latitude": 13.1000, "longitude": 77.4000, "state": "Karnataka"},
            "doddaballapur": {"latitude": 13.3000, "longitude": 77.5500, "state": "Karnataka"},
            "devanahalli": {"latitude": 13.2500, "longitude": 77.7000, "state": "Karnataka"},
            "hoskote": {"latitude": 13.0667, "longitude": 77.8000, "state": "Karnataka"},
            "bangalore": {"latitude": 12.9716, "longitude": 77.5946, "state": "Karnataka"},
            "anekal": {"latitude": 12.7000, "longitude": 77.7000, "state": "Karnataka"},
            "hoskote": {"latitude": 13.0667, "longitude": 77.8000, "state": "Karnataka"},
            "kolar": {"latitude": 13.1372, "longitude": 78.1300, "state": "Karnataka"},
            "malur": {"latitude": 12.8167, "longitude": 78.2667, "state": "Karnataka"},
            "bangarapet": {"latitude": 12.9667, "longitude": 78.2000, "state": "Karnataka"},
            "kolar gold fields": {"latitude": 12.9667, "longitude": 78.2667, "state": "Karnataka"},
            "mulbagal": {"latitude": 13.1667, "longitude": 78.4000, "state": "Karnataka"},
            "srinivaspur": {"latitude": 13.3333, "longitude": 78.2167, "state": "Karnataka"},
            "chintamani": {"latitude": 13.4000, "longitude": 78.0667, "state": "Karnataka"},
            "gudibanda": {"latitude": 13.6667, "longitude": 77.7000, "state": "Karnataka"},
            "sidlaghatta": {"latitude": 13.3833, "longitude": 77.8667, "state": "Karnataka"},
            "chikballapur": {"latitude": 13.4352, "longitude": 77.7275, "state": "Karnataka"},
            "gauribidanur": {"latitude": 13.6167, "longitude": 77.5167, "state": "Karnataka"},
            "bagepalli": {"latitude": 13.7833, "longitude": 77.7833, "state": "Karnataka"},
            "hindupur": {"latitude": 13.8333, "longitude": 77.5000, "state": "Karnataka"},
            "penukonda": {"latitude": 14.0833, "longitude": 77.5833, "state": "Karnataka"},
            "pavagada": {"latitude": 14.1000, "longitude": 77.2667, "state": "Karnataka"},
            "sira": {"latitude": 13.7500, "longitude": 76.9000, "state": "Karnataka"},
            "tumkur": {"latitude": 13.3392, "longitude": 77.1139, "state": "Karnataka"},
            "tiptur": {"latitude": 13.2500, "longitude": 76.4667, "state": "Karnataka"},
            "turuvekere": {"latitude": 13.1667, "longitude": 76.2500, "state": "Karnataka"},
            "kunigal": {"latitude": 12.9667, "longitude": 77.0167, "state": "Karnataka"},
            "madugiri": {"latitude": 13.6667, "longitude": 77.0833, "state": "Karnataka"},
            "gubbi": {"latitude": 13.3167, "longitude": 76.9500, "state": "Karnataka"},
            "koratagere": {"latitude": 13.5167, "longitude": 77.2333, "state": "Karnataka"},
            "c.n. halli": {"latitude": 13.2000, "longitude": 76.6667, "state": "Karnataka"},
            "pavagada": {"latitude": 14.1000, "longitude": 77.2667, "state": "Karnataka"},
            "sira": {"latitude": 13.7500, "longitude": 76.9000, "state": "Karnataka"},
            "tumkur": {"latitude": 13.3392, "longitude": 77.1139, "state": "Karnataka"},
            "tiptur": {"latitude": 13.2500, "longitude": 76.4667, "state": "Karnataka"},
            "turuvekere": {"latitude": 13.1667, "longitude": 76.2500, "state": "Karnataka"},
            "kunigal": {"latitude": 12.9667, "longitude": 77.0167, "state": "Karnataka"},
            "madugiri": {"latitude": 13.6667, "longitude": 77.0833, "state": "Karnataka"},
            "gubbi": {"latitude": 13.3167, "longitude": 76.9500, "state": "Karnataka"},
            "koratagere": {"latitude": 13.5167, "longitude": 77.2333, "state": "Karnataka"},
            "c.n. halli": {"latitude": 13.2000, "longitude": 76.6667, "state": "Karnataka"},
            "pavagada": {"latitude": 14.1000, "longitude": 77.2667, "state": "Karnataka"},
            "sira": {"latitude": 13.7500, "longitude": 76.9000, "state": "Karnataka"},
            "tumkur": {"latitude": 13.3392, "longitude": 77.1139, "state": "Karnataka"},
            "tiptur": {"latitude": 13.2500, "longitude": 76.4667, "state": "Karnataka"},
            "turuvekere": {"latitude": 13.1667, "longitude": 76.2500, "state": "Karnataka"},
            "kunigal": {"latitude": 12.9667, "longitude": 77.0167, "state": "Karnataka"},
            "madugiri": {"latitude": 13.6667, "longitude": 77.0833, "state": "Karnataka"},
            "gubbi": {"latitude": 13.3167, "longitude": 76.9500, "state": "Karnataka"},
            "koratagere": {"latitude": 13.5167, "longitude": 77.2333, "state": "Karnataka"},
            "c.n. halli": {"latitude": 13.2000, "longitude": 76.6667, "state": "Karnataka"},
            # Other major Indian cities
            "mumbai": {"latitude": 19.0760, "longitude": 72.8777, "state": "Maharashtra"},
            "delhi": {"latitude": 28.7041, "longitude": 77.1025, "state": "Delhi"},
            "chennai": {"latitude": 13.0827, "longitude": 80.2707, "state": "Tamil Nadu"},
            "kolkata": {"latitude": 22.5726, "longitude": 88.3639, "state": "West Bengal"},
            "hyderabad": {"latitude": 17.3850, "longitude": 78.4867, "state": "Telangana"},
            "pune": {"latitude": 18.5204, "longitude": 73.8567, "state": "Maharashtra"},
            "ahmedabad": {"latitude": 23.0225, "longitude": 72.5714, "state": "Gujarat"},
            "jaipur": {"latitude": 26.9124, "longitude": 75.7873, "state": "Rajasthan"},
            "lucknow": {"latitude": 26.8467, "longitude": 80.9462, "state": "Uttar Pradesh"}
        }
    
    def get_coordinates(self, city: str, state: str = "Karnataka", country: str = "India") -> Optional[Dict[str, float]]:
        """Get coordinates for a city with fallback to predefined coordinates."""
        try:
            # Try geocoding API first
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
                logger.info(f"Found coordinates for {city}: {result['latitude']}, {result['longitude']}")
                return {
                    "latitude": result["latitude"],
                    "longitude": result["longitude"]
                }
        except Exception as e:
            logger.warning(f"Geocoding failed for {city}: {e}")
        
        # Fallback to predefined coordinates
        city_key = city.lower().strip()
        if city_key in self.fallback_coordinates:
            coords = self.fallback_coordinates[city_key]
            logger.info(f"Using fallback coordinates for {city}: {coords}")
            return coords
        
        # Try partial matches
        for key, coords in self.fallback_coordinates.items():
            if city_key in key or key in city_key:
                logger.info(f"Using partial match coordinates for {city}: {coords}")
                return coords
        
        logger.error(f"No coordinates found for {city}")
        return None
    
    def get_karnataka_cities(self) -> List[Dict[str, str]]:
        """Get list of Karnataka cities with coordinates."""
        karnataka_cities = []
        for city_name, coords in self.fallback_coordinates.items():
            if coords.get("state") == "Karnataka":
                karnataka_cities.append({
                    "name": city_name.title(),
                    "key": city_name,
                    "state": coords["state"]
                })
        
        # Sort cities alphabetically
        karnataka_cities.sort(key=lambda x: x["name"])
        return karnataka_cities
    
    def _get_weather_description(self, weather_code: int) -> str:
        """Get weather description from weather code."""
        weather_descriptions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow fall",
            73: "Moderate snow fall",
            75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        return weather_descriptions.get(weather_code, "Unknown")
    
    def get_current_weather(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """Get current weather data with enhanced parameters."""
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,weather_code,cloud_cover,pressure_msl",
                "timezone": "auto"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data.get("current", {})
            
            weather_data = {
                "temperature": current.get("temperature_2m", 0),
                "humidity": current.get("relative_humidity_2m", 0),
                "precipitation": current.get("precipitation", 0),
                "wind_speed": current.get("wind_speed_10m", 0),
                "wind_direction": current.get("wind_direction_10m", 0),
                "weather_code": current.get("weather_code", 0),
                "cloud_cover": current.get("cloud_cover", 0),
                "pressure": current.get("pressure_msl", 0),
                "timestamp": current.get("time", datetime.now().isoformat())
            }
            
            # Add weather description based on weather code
            weather_data["description"] = self._get_weather_description(weather_data["weather_code"])
            
            logger.info(f"Weather data retrieved for {latitude}, {longitude}")
            return weather_data
            
        except Exception as e:
            logger.error(f"Could not fetch current weather: {e}")
            st.warning(f"Could not fetch current weather: {e}")
            return None
    
    def get_weather_forecast(self, latitude: float, longitude: float, days: int = 7) -> Optional[List[Dict[str, Any]]]:
        """Get weather forecast for the next few days with enhanced data."""
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code,wind_speed_10m_max,wind_direction_10m_dominant",
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
                    "precipitation_prob": daily["precipitation_probability_max"][i],
                    "weather_code": daily["weather_code"][i],
                    "description": self._get_weather_description(daily["weather_code"][i]),
                    "wind_speed_max": daily["wind_speed_10m_max"][i],
                    "wind_direction": daily["wind_direction_10m_dominant"][i]
                })
            
            logger.info(f"Weather forecast retrieved for {latitude}, {longitude} - {days} days")
            return forecast
            
        except Exception as e:
            logger.error(f"Could not fetch weather forecast: {e}")
            st.warning(f"Could not fetch weather forecast: {e}")
            return None
    
    def get_historical_weather(self, latitude: float, longitude: float, days_back: int = 7) -> Optional[List[Dict[str, Any]]]:
        """Get historical weather data for the past few days."""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max",
                "timezone": "auto"
            }
            
            response = requests.get(self.historical_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            daily = data.get("daily", {})
            
            historical = []
            for i in range(len(daily.get("time", []))):
                historical.append({
                    "date": daily["time"][i],
                    "max_temp": daily["temperature_2m_max"][i],
                    "min_temp": daily["temperature_2m_min"][i],
                    "precipitation": daily["precipitation_sum"][i],
                    "weather_code": daily["weather_code"][i],
                    "description": self._get_weather_description(daily["weather_code"][i]),
                    "wind_speed_max": daily["wind_speed_10m_max"][i]
                })
            
            logger.info(f"Historical weather data retrieved for {latitude}, {longitude} - {days_back} days")
            return historical
            
        except Exception as e:
            logger.error(f"Could not fetch historical weather: {e}")
            st.warning(f"Could not fetch historical weather: {e}")
            return None
    
    def get_weather_alerts(self, latitude: float, longitude: float) -> List[Dict[str, Any]]:
        """Get weather alerts based on current and forecast conditions."""
        alerts = []
        
        try:
            # Get current weather and forecast
            current_weather = self.get_current_weather(latitude, longitude)
            forecast = self.get_weather_forecast(latitude, longitude, 3)
            
            if current_weather:
                # Temperature alerts
                temp = current_weather["temperature"]
                if temp > 40:
                    alerts.append({
                        "type": "heat",
                        "severity": "critical",
                        "message": f"Extreme heat warning ({temp}°C). Immediate irrigation required.",
                        "icon": "🔥",
                        "priority": "high"
                    })
                elif temp > 35:
                    alerts.append({
                        "type": "heat",
                        "severity": "high",
                        "message": f"High temperature alert ({temp}°C). Increase irrigation frequency.",
                        "icon": "🌡️",
                        "priority": "medium"
                    })
                elif temp < 5:
                    alerts.append({
                        "type": "cold",
                        "severity": "high",
                        "message": f"Frost warning ({temp}°C). Protect sensitive crops immediately.",
                        "icon": "❄️",
                        "priority": "high"
                    })
                elif temp < 15:
                    alerts.append({
                        "type": "cold",
                        "severity": "moderate",
                        "message": f"Low temperature ({temp}°C). Monitor crop health.",
                        "icon": "🧊",
                        "priority": "low"
                    })
                
                # Precipitation alerts
                precipitation = current_weather["precipitation"]
                if precipitation > 20:
                    alerts.append({
                        "type": "rain",
                        "severity": "high",
                        "message": f"Heavy rainfall ({precipitation}mm). Ensure proper drainage.",
                        "icon": "🌧️",
                        "priority": "high"
                    })
                elif precipitation > 10:
                    alerts.append({
                        "type": "rain",
                        "severity": "moderate",
                        "message": f"Moderate rainfall ({precipitation}mm). Monitor soil moisture.",
                        "icon": "🌦️",
                        "priority": "medium"
                    })
                
                # Wind alerts
                wind_speed = current_weather["wind_speed"]
                if wind_speed > 25:
                    alerts.append({
                        "type": "wind",
                        "severity": "high",
                        "message": f"Strong winds ({wind_speed} km/h). Secure all equipment and protect crops.",
                        "icon": "💨",
                        "priority": "high"
                    })
                elif wind_speed > 15:
                    alerts.append({
                        "type": "wind",
                        "severity": "moderate",
                        "message": f"Moderate winds ({wind_speed} km/h). Protect young plants.",
                        "icon": "🌬️",
                        "priority": "medium"
                    })
            
            # Forecast-based alerts
            if forecast:
                # Check for upcoming extreme weather
                for day in forecast[:2]:  # Next 2 days
                    max_temp = day.get("max_temp", 0)
                    precipitation = day.get("precipitation", 0)
                    
                    if max_temp > 40:
                        alerts.append({
                            "type": "forecast_heat",
                            "severity": "high",
                            "message": f"Extreme heat forecast for {day['date']} ({max_temp}°C). Prepare irrigation.",
                            "icon": "🔥",
                            "priority": "high"
                        })
                    
                    if precipitation > 15:
                        alerts.append({
                            "type": "forecast_rain",
                            "severity": "moderate",
                            "message": f"Heavy rain forecast for {day['date']} ({precipitation}mm). Check drainage.",
                            "icon": "🌧️",
                            "priority": "medium"
                        })
            
            logger.info(f"Weather alerts generated: {len(alerts)} alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating weather alerts: {e}")
            return []
    
    def get_comprehensive_weather_data(self, city: str, state: str = "Karnataka", country: str = "India") -> Optional[Dict[str, Any]]:
        """Get comprehensive weather data including current, forecast, historical, and alerts."""
        coords = self.get_coordinates(city, state, country)
        if not coords:
            return None
        
        try:
            current = self.get_current_weather(coords["latitude"], coords["longitude"])
            forecast = self.get_weather_forecast(coords["latitude"], coords["longitude"], 7)
            historical = self.get_historical_weather(coords["latitude"], coords["longitude"], 7)
            alerts = self.get_weather_alerts(coords["latitude"], coords["longitude"])
            
            return {
                "location": {
                    "city": city,
                    "state": state,
                    "country": country,
                    "coordinates": coords
                },
                "current": current,
                "forecast": forecast,
                "historical": historical,
                "alerts": alerts,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive weather data: {e}")
            return None

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

def get_weather_recommendations(weather_data: Dict[str, Any], crop_type: str) -> List[Dict[str, Any]]:
    """Get weather-based recommendations for farming with enhanced analysis."""
    recommendations = []
    
    if not weather_data:
        return recommendations
    
    current = weather_data.get("current", {})
    forecast = weather_data.get("forecast", [])
    historical = weather_data.get("historical", [])
    
    # Temperature recommendations
    temp = current.get("temperature", 0)
    if temp > 40:
        recommendations.append({
            "type": "irrigation",
            "priority": "critical",
            "message": f"Extreme heat ({temp}°C) detected. Immediate irrigation required to prevent crop damage.",
            "icon": "🔥",
            "action": "Increase irrigation frequency to 2-3 times daily"
        })
    elif temp > 35:
        recommendations.append({
            "type": "irrigation",
            "priority": "high",
            "message": f"High temperature ({temp}°C) detected. Increase irrigation frequency.",
            "icon": "🌡️",
            "action": "Water crops early morning and evening"
        })
    elif temp < 5:
        recommendations.append({
            "type": "protection",
            "priority": "critical",
            "message": f"Frost warning ({temp}°C). Protect sensitive crops immediately.",
            "icon": "❄️",
            "action": "Cover crops with frost cloth or mulch"
        })
    elif temp < 15:
        recommendations.append({
            "type": "protection",
            "priority": "medium",
            "message": f"Low temperature ({temp}°C) detected. Monitor crop health.",
            "icon": "🧊",
            "action": "Check for frost damage and consider covering sensitive plants"
        })
    
    # Precipitation recommendations
    precipitation = current.get("precipitation", 0)
    if precipitation > 20:
        recommendations.append({
            "type": "drainage",
            "priority": "high",
            "message": f"Heavy rainfall ({precipitation}mm). Ensure proper drainage.",
            "icon": "🌧️",
            "action": "Check and clear drainage channels immediately"
        })
    elif precipitation > 10:
        recommendations.append({
            "type": "drainage",
            "priority": "medium",
            "message": f"Moderate rainfall ({precipitation}mm). Monitor soil moisture.",
            "icon": "🌦️",
            "action": "Check soil drainage and avoid overwatering"
        })
    elif precipitation < 1 and any(day.get("precipitation", 0) < 1 for day in forecast[:3]):
        recommendations.append({
            "type": "irrigation",
            "priority": "medium",
            "message": "Dry conditions expected. Plan irrigation schedule.",
            "icon": "💧",
            "action": "Schedule regular irrigation based on soil moisture"
        })
    
    # Wind recommendations
    wind_speed = current.get("wind_speed", 0)
    if wind_speed > 25:
        recommendations.append({
            "type": "protection",
            "priority": "high",
            "message": f"Strong winds ({wind_speed} km/h). Secure all equipment and protect crops.",
            "icon": "💨",
            "action": "Secure trellises, greenhouses, and protect young plants"
        })
    elif wind_speed > 15:
        recommendations.append({
            "type": "protection",
            "priority": "medium",
            "message": f"Moderate winds ({wind_speed} km/h). Protect young plants.",
            "icon": "🌬️",
            "action": "Provide windbreaks or temporary protection for seedlings"
        })
    
    # Humidity recommendations
    humidity = current.get("humidity", 0)
    if humidity > 80:
        recommendations.append({
            "type": "disease_prevention",
            "priority": "medium",
            "message": f"High humidity ({humidity}%) increases disease risk. Monitor for fungal infections.",
            "icon": "🦠",
            "action": "Improve air circulation and consider fungicide application"
        })
    elif humidity < 30:
        recommendations.append({
            "type": "irrigation",
            "priority": "medium",
            "message": f"Low humidity ({humidity}%) increases water stress. Monitor soil moisture.",
            "icon": "💧",
            "action": "Increase irrigation frequency and consider mulching"
        })
    
    # Crop-specific recommendations
    crop_lower = crop_type.lower()
    if crop_lower == "rice":
        if precipitation > 5:
            recommendations.append({
                "type": "crop_specific",
                "priority": "low",
                "message": "Good rainfall for rice cultivation. Monitor water levels.",
                "icon": "🌾",
                "action": "Maintain 5-10cm water depth in rice fields"
            })
        if temp > 35:
            recommendations.append({
                "type": "crop_specific",
                "priority": "medium",
                "message": "High temperature may affect rice flowering. Monitor closely.",
                "icon": "🌾",
                "action": "Ensure adequate water supply during flowering stage"
            })
    elif crop_lower == "wheat":
        if temp > 30:
            recommendations.append({
                "type": "crop_specific",
                "priority": "medium",
                "message": "High temperature may affect wheat grain filling. Monitor closely.",
                "icon": "🌾",
                "action": "Apply light irrigation to cool the crop canopy"
            })
        if temp < 10:
            recommendations.append({
                "type": "crop_specific",
                "priority": "medium",
                "message": "Low temperature may slow wheat growth. Consider protection.",
                "icon": "🌾",
                "action": "Monitor for frost damage and consider light irrigation"
            })
    elif crop_lower == "tomato":
        if humidity > 70:
            recommendations.append({
                "type": "crop_specific",
                "priority": "high",
                "message": "High humidity increases tomato disease risk. Monitor for blight.",
                "icon": "🍅",
                "action": "Improve air circulation and check for early blight symptoms"
            })
        if temp > 35:
            recommendations.append({
                "type": "crop_specific",
                "priority": "medium",
                "message": "High temperature may cause tomato flower drop.",
                "icon": "🍅",
                "action": "Provide shade and ensure adequate water supply"
            })
    
    # Historical trend analysis
    if historical and len(historical) >= 3:
        recent_temps = [day.get("max_temp", 0) for day in historical[-3:] if day.get("max_temp") is not None]
        if recent_temps:
            avg_temp = sum(recent_temps) / len(recent_temps)
            
            if avg_temp > temp + 5:
                recommendations.append({
                    "type": "trend",
                    "priority": "low",
                    "message": f"Temperature trend shows cooling (avg: {avg_temp:.1f}°C vs current: {temp}°C).",
                    "icon": "📉",
                    "action": "Monitor for potential cold stress in sensitive crops"
                })
            elif avg_temp < temp - 5:
                recommendations.append({
                    "type": "trend",
                    "priority": "low",
                    "message": f"Temperature trend shows warming (avg: {avg_temp:.1f}°C vs current: {temp}°C).",
                    "icon": "📈",
                    "action": "Prepare for increased irrigation needs"
                })
    
    return recommendations
