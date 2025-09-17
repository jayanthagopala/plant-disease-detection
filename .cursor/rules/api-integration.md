---
type: Auto Attached
globs: ["**/api/**", "**/*api*", "**/services/**"]
description: Guidelines for integrating free agriculture APIs
---

# API Integration Guidelines

## Free API Endpoints
1. **Weather**: Open-Meteo (no API key required)
2. **Soil Data**: SoilGrids REST API
3. **Market Prices**: Commodities-API (free tier)
4. **Geocoding**: OpenStreetMap Nominatim

## API Integration Patterns
```typescript
// Always implement proper error handling and fallbacks
const fetchWeatherData = async (lat: number, lon: number) => {
  try {
    const response = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,precipitation`
    );
    if (!response.ok) throw new Error(`Weather API error: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error('Weather API failed:', error);
    // Return fallback data or cached data
    return getFallbackWeatherData();
  }
};
```

## Rate Limiting & Caching
- Implement request caching for 5-15 minutes
- Add exponential backoff for failures
- Use localStorage for offline data persistence
- Implement request deduplication

## Error Handling
- Always provide fallback data
- Display user-friendly error messages in local language
- Log errors for monitoring
- Implement retry mechanisms with limits
