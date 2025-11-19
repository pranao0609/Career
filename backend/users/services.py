import logging
from datetime import datetime
from typing import Optional, Dict, Any
from database.dependencies import get_database, get_original_database

logger = logging.getLogger(__name__)

class UserService:
    """Service class for user operations"""
    
    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            try:
                database = get_database()
            except:
                database = get_original_database()
            
            user_doc = database.collection("users").document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                user_data["_id"] = user_doc.id
                user_data.pop("password", None)
                return user_data
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    @staticmethod
    def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            try:
                database = get_database()
            except:
                database = get_original_database()
            
            users = database.collection("users").where("email", "==", email).get()
            if users:
                user_data = users[0].to_dict()
                user_data["_id"] = users[0].id
                user_data.pop("password", None)
                return user_data
            return None
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    @staticmethod
    def update_user_profile(user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user profile"""
        try:
            try:
                database = get_database()
            except:
                database = get_original_database()
            
            update_data["updated_at"] = datetime.utcnow()
            database.collection("users").document(user_id).update(update_data)
            return True
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False
    
    @staticmethod
    def update_last_activity(user_id: str, status: str = "online") -> bool:
        """Update user's last activity"""
        try:
            try:
                database = get_database()
            except:
                database = get_original_database()
            
            database.collection("users").document(user_id).update({
                "last_activity": datetime.utcnow(),
                "status": status,
                "updated_at": datetime.utcnow()
            })
            return True
        except Exception as e:
            logger.error(f"Error updating last activity: {e}")
            return False