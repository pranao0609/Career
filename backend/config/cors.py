import os
from typing import List
from config.settings import get_settings

def get_cors_origins() -> List[str]:
    """Get CORS origins from environment and defaults"""
    settings = get_settings()
    
    # Default origins
    origins = [
        "http://localhost:8080",  # React frontend port
        "http://127.0.0.1:8080",
        "http://localhost:3000",  # Alternative React port
    ]
    
    # Add development wildcard
    if settings.DEBUG:
        origins.append("*")
    
    # Add environment origins
    try:
        env_origins = settings.CORS_ORIGINS.split(",")
        if env_origins and env_origins[0]:  # If environment origins exist
            origins.extend([origin.strip() for origin in env_origins])
    except Exception:
        pass
    
    return origins