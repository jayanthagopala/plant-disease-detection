"""Market price tracking service for the Smart Crop Advisory System."""

import requests
import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

class MarketPriceService:
    """Market price service using mock data and free APIs."""
    
    def __init__(self):
        self.base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        self.api_key = "579b464db66ec23bdd000001"  # Free API key for Indian government data
        
    def get_crop_prices(self, state: str = "Karnataka", district: str = "Bangalore") -> Optional[List[Dict[str, Any]]]:
        """Get current crop prices from government API."""
        try:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "filters[state]": state,
                "filters[district]": district,
                "limit": 50
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            records = data.get("records", [])
            
            # Process and format the data
            prices = []
            for record in records:
                prices.append({
                    "commodity": record.get("commodity", ""),
                    "variety": record.get("variety", ""),
                    "market": record.get("market", ""),
                    "min_price": float(record.get("min_price", 0)),
                    "max_price": float(record.get("max_price", 0)),
                    "modal_price": float(record.get("modal_price", 0)),
                    "date": record.get("arrival_date", ""),
                    "state": record.get("state", ""),
                    "district": record.get("district", "")
                })
            
            return prices
        except Exception as e:
            st.warning(f"Could not fetch market prices: {e}")
            return None
    
    def get_price_trends(self, commodity: str, days: int = 30) -> Optional[List[Dict[str, Any]]]:
        """Get price trends for a specific commodity (mock implementation)."""
        # In a real implementation, this would fetch historical data
        # For demo purposes, we'll generate mock trend data
        trends = []
        base_price = self._get_base_price(commodity)
        
        for i in range(days):
            # Simulate price fluctuations
            variation = (i % 7 - 3) * 0.05  # Weekly cycle
            random_factor = (hash(f"{commodity}{i}") % 100 - 50) / 1000  # Random variation
            price = base_price * (1 + variation + random_factor)
            
            trends.append({
                "date": (datetime.now() - timedelta(days=days-i)).strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "change": round((price - base_price) / base_price * 100, 2)
            })
        
        return trends
    
    def _get_base_price(self, commodity: str) -> float:
        """Get base price for a commodity."""
        base_prices = {
            "rice": 2500,
            "wheat": 2200,
            "maize": 1800,
            "tomato": 3000,
            "potato": 1500,
            "onion": 2000,
            "sugarcane": 300,
            "cotton": 6000
        }
        return base_prices.get(commodity.lower(), 2000)
    
    def get_market_recommendations(self, prices: List[Dict[str, Any]], crop_type: str) -> List[Dict[str, Any]]:
        """Get market-based recommendations for selling crops."""
        recommendations = []
        
        if not prices:
            return recommendations
        
        # Find prices for the specific crop
        crop_prices = [p for p in prices if crop_type.lower() in p.get("commodity", "").lower()]
        
        if not crop_prices:
            return recommendations
        
        # Calculate average price
        avg_price = sum(p["modal_price"] for p in crop_prices) / len(crop_prices)
        
        # Get price trends
        trends = self.get_price_trends(crop_type, 7)
        if trends:
            recent_trend = trends[-1]["change"]
            
            if recent_trend > 5:
                recommendations.append({
                    "type": "sell",
                    "priority": "high",
                    "message": f"Price increased by {recent_trend:.1f}%. Consider selling now for better profit.",
                    "icon": "📈",
                    "price": avg_price
                })
            elif recent_trend < -5:
                recommendations.append({
                    "type": "hold",
                    "priority": "medium",
                    "message": f"Price decreased by {abs(recent_trend):.1f}%. Consider waiting for better prices.",
                    "icon": "📉",
                    "price": avg_price
                })
            else:
                recommendations.append({
                    "type": "monitor",
                    "priority": "low",
                    "message": f"Price stable. Current average: ₹{avg_price:.0f}/quintal. Monitor market trends.",
                    "icon": "📊",
                    "price": avg_price
                })
        
        return recommendations

def create_mock_market_data() -> Dict[str, Any]:
    """Create mock market data for demonstration."""
    return {
        "prices": [
            {
                "commodity": "Rice",
                "variety": "Basmati",
                "market": "Mysore",
                "min_price": 2800,
                "max_price": 3200,
                "modal_price": 3000,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "state": "Karnataka",
                "district": "Mysore"
            },
            {
                "commodity": "Wheat",
                "variety": "Durum",
                "market": "Bangalore",
                "min_price": 2000,
                "max_price": 2400,
                "modal_price": 2200,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "state": "Karnataka",
                "district": "Bangalore"
            },
            {
                "commodity": "Tomato",
                "variety": "Hybrid",
                "market": "Mysore",
                "min_price": 2500,
                "max_price": 3500,
                "modal_price": 3000,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "state": "Karnataka",
                "district": "Mysore"
            },
            {
                "commodity": "Potato",
                "variety": "Kufri",
                "market": "Bangalore",
                "min_price": 1200,
                "max_price": 1800,
                "modal_price": 1500,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "state": "Karnataka",
                "district": "Bangalore"
            }
        ],
        "trends": {
            "rice": [
                {"date": "2024-01-01", "price": 2800, "change": 2.5},
                {"date": "2024-01-02", "price": 2850, "change": 1.8},
                {"date": "2024-01-03", "price": 2900, "change": 1.7},
                {"date": "2024-01-04", "price": 2950, "change": 1.7},
                {"date": "2024-01-05", "price": 3000, "change": 1.7}
            ],
            "wheat": [
                {"date": "2024-01-01", "price": 2100, "change": -1.2},
                {"date": "2024-01-02", "price": 2150, "change": 2.4},
                {"date": "2024-01-03", "price": 2180, "change": 1.4},
                {"date": "2024-01-04", "price": 2200, "change": 0.9},
                {"date": "2024-01-05", "price": 2200, "change": 0.0}
            ]
        }
    }

def get_market_insights(prices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get market insights and analysis."""
    if not prices:
        return {"message": "No market data available"}
    
    # Calculate market statistics
    all_prices = [p["modal_price"] for p in prices]
    avg_price = sum(all_prices) / len(all_prices)
    min_price = min(all_prices)
    max_price = max(all_prices)
    
    # Find best and worst performing crops
    price_changes = []
    for price in prices:
        # Mock price change calculation
        change = (hash(price["commodity"]) % 20 - 10) / 100  # Random change between -10% and +10%
        price_changes.append({
            "commodity": price["commodity"],
            "change": change,
            "price": price["modal_price"]
        })
    
    best_performer = max(price_changes, key=lambda x: x["change"])
    worst_performer = min(price_changes, key=lambda x: x["change"])
    
    return {
        "average_price": avg_price,
        "price_range": {"min": min_price, "max": max_price},
        "best_performer": best_performer,
        "worst_performer": worst_performer,
        "total_commodities": len(prices),
        "market_activity": "High" if len(prices) > 10 else "Moderate"
    }
