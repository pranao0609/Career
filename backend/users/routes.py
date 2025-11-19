from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import logging

from users.models import UserProfile, UserChatInfo
from users.services import UserService
from auth.dependencies import require_authentication
from database.dependencies import get_database, get_original_database

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/profile")
def get_profile_enhanced(current_user: dict = Depends(require_authentication)):
    """Get current user profile (enhanced version)"""
    profile = current_user.copy()
    profile.pop("password", None)
    
    return {
        "user_id": profile.get("user_id", profile.get("email")),
        "firstName": profile.get("firstName", ""),
        "lastName": profile.get("lastName", ""),
        "email": profile["email"],
        "avatar_url": profile.get("avatar_url"),
        "bio": profile.get("bio", ""),
        "status": profile.get("status", "online"),
        "chat_settings": profile.get("chat_settings", {}),
        "preferences": profile.get("preferences", {})
    }

@router.put("/profile")
def update_profile(
    profile_data: UserProfile,
    current_user: dict = Depends(require_authentication)
):
    """Update user profile"""
    try:
        # Try enhanced database first
        try:
            database = get_database()
        except:
            database = get_original_database()
        
        user_ref = database.collection("users").document(current_user["_id"])
        
        update_data = {"updated_at": datetime.utcnow()}
        
        # Update only provided fields
        for field, value in profile_data.dict(exclude_unset=True).items():
            update_data[field] = value
            
        user_ref.update(update_data)
        
        return {"message": "Profile updated successfully"}
        
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@router.get("/chat-info")
def get_chat_info(current_user: dict = Depends(require_authentication)):
    """Get user information for chat"""
    return {
        "user_id": current_user.get("user_id", current_user["email"]),
        "username": f"{current_user.get('firstName', '')} {current_user.get('lastName', '')}".strip(),
        "email": current_user["email"],
        "avatar_url": current_user.get("avatar_url"),
        "status": current_user.get("status", "online"),
        "chat_settings": current_user.get("chat_settings", {}),
        "is_online": True
    }

@router.put("/status")
def update_status(
    status_data: dict,
    current_user: dict = Depends(require_authentication)
):
    """Update user status"""
    try:
        status = status_data.get("status", "online")
        if status not in ["online", "offline", "away", "busy"]:
            raise HTTPException(status_code=400, detail="Invalid status")
        
        success = UserService.update_last_activity(current_user["_id"], status)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update status")
        
        return {"message": "Status updated successfully", "status": status}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update status")

@router.get("/settings")
def get_user_settings(current_user: dict = Depends(require_authentication)):
    """Get user settings"""
    return {
        "chat_settings": current_user.get("chat_settings", {}),
        "preferences": current_user.get("preferences", {}),
        "privacy": current_user.get("privacy", {}),
        "notifications": current_user.get("notifications", {})
    }

@router.put("/settings")
def update_user_settings(
    settings_data: dict,
    current_user: dict = Depends(require_authentication)
):
    """Update user settings"""
    try:
        # Try enhanced database first
        try:
            database = get_database()
        except:
            database = get_original_database()
        
        user_ref = database.collection("users").document(current_user["_id"])
        
        update_data = {"updated_at": datetime.utcnow()}
        
        # Update specific settings sections
        allowed_sections = ["chat_settings", "preferences", "privacy", "notifications"]
        for section in allowed_sections:
            if section in settings_data:
                update_data[section] = settings_data[section]
        
        user_ref.update(update_data)
        
        return {"message": "Settings updated successfully"}
        
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update settings")