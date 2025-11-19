from pydantic import BaseModel
from typing import Optional

class UserProfile(BaseModel):
    """User profile model"""
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    status: Optional[str] = None

class ChatSettings(BaseModel):
    """User chat settings model"""
    notifications_enabled: Optional[bool] = None
    sound_enabled: Optional[bool] = None
    online_status_visible: Optional[bool] = None

class UserPreferences(BaseModel):
    """User preferences model"""
    theme: Optional[str] = None
    language: Optional[str] = None

class UserChatInfo(BaseModel):
    """User chat information model"""
    user_id: str
    username: str
    email: str
    avatar_url: Optional[str] = None
    status: str = "online"
    chat_settings: dict = {}
    is_online: bool = True