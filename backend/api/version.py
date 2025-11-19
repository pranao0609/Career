import os
from fastapi import APIRouter
import firebase_admin
from config.settings import get_settings

router = APIRouter()
settings = get_settings()

@router.get("/version")
def get_version():
    """Get API version information"""
    return {
        "version": "1.0.0",
        "name": "Student Advisor Portal",
        "description": "Community chat platform with Firestore backend",
        "firebase_initialized": bool(firebase_admin._apps),
        "enhanced_db_available": True,
        "chat_enabled": True,
        "environment": settings.ENVIRONMENT,
        "debug_mode": settings.DEBUG,
        "python_version": os.sys.version,
        "features": {
            "authentication": True,
            "user_management": True,
            "chat_system": True,
            "career_guidance": True,
            "real_time_messaging": True,
            "google_oauth": True
        }
    }