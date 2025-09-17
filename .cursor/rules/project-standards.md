---
type: Always
description: Core development standards for the Smart Crop Advisory System
---

# Smart Crop Advisory System - Development Standards

## Project Overview
Building a multilingual AI-based crop advisory system for small and marginal farmers in India.

## Technology Stack
- **Frontend**: Streamlit (Python-based web framework)
- **Backend**: Python FastAPI or direct Streamlit integration
- **Database**: PostgreSQL with PostGIS for location data (optional for MVP)
- **AI/ML**: PyTorch with torchvision for image recognition
- **APIs**: Free-tier APIs only (Open-Meteo, SoilGrids, Commodities-API)

## Core Principles
- **Mobile-first design** - 86% of Indian farmers use mobile devices
- **Multilingual support** - Primary: Hindi, English; Future: Regional languages
- **Offline capability** - Essential for rural areas with poor connectivity
- **Free/low-cost APIs** - Sustainable for small-scale farming
- **Accessibility** - Simple UI for low-literate users
- **Data privacy** - Farmer data protection is critical

## Code Standards
- Use Python type hints for type safety
- Implement proper error handling for API failures
- Add loading states for all async operations using st.spinner()
- Use Streamlit session state for data management
- Implement proper form validation with st.form()
- Add comprehensive error handling with try-catch blocks
- Use Streamlit's built-in accessibility features
