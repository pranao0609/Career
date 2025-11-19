import logging
from datetime import datetime
from typing import Optional, Dict, Any
from google.cloud.firestore import DocumentSnapshot
from google.cloud import firestore

from .firestore import firestore_db
from models.profile import UserProfile, ProfileResponse, AcademicBackground

logger = logging.getLogger(__name__)

class ProfileService:
    def __init__(self):
        self.collection_name = "user_profiles"
    
    def _get_db(self):
        """Get Firestore database instance"""
        return firestore_db.get_db()
    
    def _serialize_profile(self, profile: UserProfile) -> Dict[str, Any]:
        """Convert Pydantic model to dict for Firestore storage"""
        return profile.model_dump(by_alias=True)
    
    def _deserialize_profile(self, data: Dict[str, Any]) -> UserProfile:
        """Convert Firestore data to Pydantic model"""
        return UserProfile.model_validate(data)
    
    def _convert_timestamp(self, timestamp_value):
        """Convert Firestore timestamp to datetime, handling various types"""
        if timestamp_value is None:
            return datetime.utcnow()
        
        # Handle Firestore timestamp objects
        if hasattr(timestamp_value, 'timestamp'):
            return datetime.fromtimestamp(timestamp_value.timestamp())
        
        # Handle datetime objects
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        
        # Handle server timestamp sentinel
        if str(timestamp_value) == 'SERVER_TIMESTAMP':
            return datetime.utcnow()
            
        # Default fallback
        return datetime.utcnow()
    
    async def create_or_update_profile(self, user_id: str, profile: UserProfile) -> ProfileResponse:
        """Create a new profile or update existing one - unified method"""
        try:
            db = self._get_db()
            doc_ref = db.collection(self.collection_name).document(user_id)
            
            # Check if profile already exists
            existing_doc = doc_ref.get()
            
            if existing_doc.exists:
                # Profile exists, update it
                logger.info(f"Profile exists for user {user_id}, updating...")
                existing_data = existing_doc.to_dict()
                created_at = self._convert_timestamp(existing_data.get("created_at"))
                now = datetime.utcnow()
                
                profile_data = {
                    "user_id": user_id,
                    "profile": self._serialize_profile(profile),
                    "created_at": created_at,
                    "updated_at": now
                }
            else:
                # Profile doesn't exist, create new one
                logger.info(f"Creating new profile for user {user_id}")
                now = datetime.utcnow()
                
                profile_data = {
                    "user_id": user_id,
                    "profile": self._serialize_profile(profile),
                    "created_at": now,
                    "updated_at": now
                }
            
            # Save to Firestore (will create or update)
            doc_ref.set(profile_data)
            
            logger.info(f"Profile saved successfully for user {user_id}")
            
            return ProfileResponse(
                user_id=user_id,
                profile=profile,
                created_at=profile_data["created_at"],
                updated_at=profile_data["updated_at"]
            )
            
        except Exception as e:
            logger.error(f"Failed to save profile for user {user_id}: {str(e)}")
            raise Exception(f"Failed to save profile: {str(e)}")

    async def create_profile(self, user_id: str, profile: UserProfile) -> ProfileResponse:
        """Create a new user profile"""
        return await self.create_or_update_profile(user_id, profile)
    
    async def get_profile(self, user_id: str) -> Optional[ProfileResponse]:
        """Get user profile by user ID"""
        try:
            db = self._get_db()
            doc_ref = db.collection(self.collection_name).document(user_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                logger.warning(f"Profile not found for user {user_id}")
                return None
            
            data = doc.to_dict()
            
            # Convert timestamps safely
            created_at = self._convert_timestamp(data.get("created_at"))
            updated_at = self._convert_timestamp(data.get("updated_at"))
            
            profile = self._deserialize_profile(data["profile"])
            
            logger.info(f"Profile retrieved for user {user_id}")
            
            return ProfileResponse(
                user_id=user_id,
                profile=profile,
                created_at=created_at,
                updated_at=updated_at
            )
            
        except Exception as e:
            logger.error(f"Failed to get profile for user {user_id}: {str(e)}")
            raise Exception(f"Failed to retrieve profile: {str(e)}")
    
    async def update_profile(self, user_id: str, profile: UserProfile) -> ProfileResponse:
        """Update user profile"""
        try:
            db = self._get_db()
            doc_ref = db.collection(self.collection_name).document(user_id)
            
            # Get existing document to preserve created_at
            existing_doc = doc_ref.get()
            created_at = datetime.utcnow()  # Default fallback
            
            if existing_doc.exists:
                existing_data = existing_doc.to_dict()
                created_at = self._convert_timestamp(existing_data.get("created_at"))
            
            # Prepare updated data
            now = datetime.utcnow()
            profile_data = {
                "user_id": user_id,
                "profile": self._serialize_profile(profile),
                "created_at": created_at,
                "updated_at": now
            }
            
            # Update in Firestore (this will create if doesn't exist)
            doc_ref.set(profile_data)
            
            logger.info(f"Profile updated for user {user_id}")
            
            return ProfileResponse(
                user_id=user_id,
                profile=profile,
                created_at=created_at,
                updated_at=now
            )
            
        except Exception as e:
            logger.error(f"Failed to update profile for user {user_id}: {str(e)}")
            raise Exception(f"Failed to update profile: {str(e)}")
    
    # NEW: Method to update only academic background
    async def update_academic_background(self, user_id: str, academic_background: AcademicBackground) -> ProfileResponse:
        """Update only the academic background portion of the profile"""
        try:
            # Get existing profile
            existing_profile = await self.get_profile(user_id)
            if not existing_profile:
                raise ValueError(f"Profile not found for user {user_id}")
            
            # Update only academic background
            updated_profile = existing_profile.profile.model_copy()
            updated_profile.academic_background = academic_background
            
            # Save updated profile
            return await self.update_profile(user_id, updated_profile)
            
        except Exception as e:
            logger.error(f"Failed to update academic background for user {user_id}: {str(e)}")
            raise Exception(f"Failed to update academic background: {str(e)}")
    
    # NEW: Method to get only academic background
    async def get_academic_background(self, user_id: str) -> Optional[AcademicBackground]:
        """Get only the academic background portion"""
        try:
            profile_response = await self.get_profile(user_id)
            if not profile_response:
                return None
            
            return profile_response.profile.academic_background
            
        except Exception as e:
            logger.error(f"Failed to get academic background for user {user_id}: {str(e)}")
            return None
    
    async def delete_profile(self, user_id: str) -> bool:
        """Delete user profile"""
        try:
            db = self._get_db()
            doc_ref = db.collection(self.collection_name).document(user_id)
            
            # Check if document exists
            doc = doc_ref.get()
            if not doc.exists:
                logger.warning(f"Profile not found for deletion: {user_id}")
                return False
            
            # Delete the document
            doc_ref.delete()
            
            logger.info(f"Profile deleted for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete profile for user {user_id}: {str(e)}")
            raise Exception(f"Failed to delete profile: {str(e)}")
    
    async def profile_exists(self, user_id: str) -> bool:
        """Check if user profile exists"""
        try:
            db = self._get_db()
            doc_ref = db.collection(self.collection_name).document(user_id)
            doc = doc_ref.get()
            
            return doc.exists
            
        except Exception as e:
            logger.error(f"Failed to check profile existence for user {user_id}: {str(e)}")
            return False

# Global profile service instance
profile_service = ProfileService()