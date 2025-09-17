---
type: Auto Attached
globs: ["**/i18n/**", "**/locales/**", "**/*translation*"]
description: Multilingual support for Indian farmers
---

# Localization Guidelines

## Supported Languages
- **Primary**: Hindi (हिंदी), English  
- **Future**: Bengali, Telugu, Tamil, Gujarati, Marathi

## Translation File Structure
```json
// en/common.json
{
  "navigation": {
    "home": "Home",
    "weather": "Weather", 
    "soil": "Soil Health",
    "disease": "Disease Detection",
    "prices": "Market Prices",
    "chat": "Advisory Chat"
  },
  "actions": {
    "save": "Save",
    "cancel": "Cancel",
    "upload": "Upload Image",
    "analyze": "Analyze",
    "send": "Send"
  }
}

// hi/common.json  
{
  "navigation": {
    "home": "होम",
    "weather": "मौसम",
    "soil": "मिट्टी का स्वास्थ्य", 
    "disease": "रोग की पहचान",
    "prices": "बाजार भाव",
    "chat": "सलाह चैट"
  }
}
```

## Agricultural Terminology
- Use locally understood farming terms
- Include crop names in regional languages
- Provide audio pronunciations for key terms
- Use metric system (hectares, quintals)

## Cultural Considerations
- Use appropriate colors (green for good, red for warnings)
- Include local farming calendar context
- Respect regional farming practices
- Consider monsoon season terminology
