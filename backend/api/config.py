import os
from fastapi import APIRouter
from config.settings import get_settings

router = APIRouter()
settings = get_settings()

@router.get("/config")
def get_config():
    """Get client configuration"""
    return {
        "websocket_url": f"ws://localhost:{settings.PORT}/api/chat/ws",
        "api_base_url": f"http://localhost:{settings.PORT}",
        "features": {
            "chat": True,
            "google_auth": True,
            "manual_auth": True,
            "file_upload": False,
            "notifications": True
        },
        "limits": {
            "message_length": settings.MESSAGE_LENGTH_LIMIT,
            "room_members": settings.ROOM_MEMBERS_LIMIT,
            "file_size_mb": settings.FILE_SIZE_LIMIT_MB
        },
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG
    }