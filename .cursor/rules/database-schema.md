---
type: Auto Attached
globs: ["**/db/**", "**/*model*", "**/*schema*", "**/*.sql"]
description: Database design for agricultural data
---

# Database Schema Guidelines

## Core Tables Structure
```sql
-- Users table with location and preferences
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_number VARCHAR(15) UNIQUE,
    preferred_language VARCHAR(10) DEFAULT 'hindi',
    location GEOGRAPHY(POINT, 4326),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Crops and farming data
CREATE TABLE user_crops (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    crop_type VARCHAR(50),
    planting_date DATE,
    expected_harvest DATE,
    field_size_acres DECIMAL(6,2)
);

-- Weather data cache
CREATE TABLE weather_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location GEOGRAPHY(POINT, 4326),
    weather_data JSONB,
    cached_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- Disease detection history
CREATE TABLE disease_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    image_url TEXT,
    detected_disease VARCHAR(100),
    confidence_score DECIMAL(3,2),
    treatment_advice JSONB,
    detected_at TIMESTAMP DEFAULT NOW()
);
```

## Indexing Strategy
- Add spatial indexes for location-based queries
- Index frequently queried fields (user_id, created_at)
- Use partial indexes for active crops only

## Data Privacy
- Encrypt sensitive farmer data
- Implement data retention policies
- Allow users to delete their data completely
- Don't store images permanently - process and delete
