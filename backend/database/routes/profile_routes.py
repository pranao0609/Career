from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Optional

from models.profile import UserProfile, ProfileResponse, UpdateProfileRequest
from ..profile_service import profile_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/profile", tags=["profile"])

# For now, we'll use a simple user_id parameter. 
# In a real app, you'd get this from JWT token or session
async def get_current_user_id(user_id: str) -> str:
    """
    Placeholder for user authentication.
    In production, replace this with proper JWT token validation.
    """
    if not user_id or len(user_id.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID is required"
        )
    return user_id.strip()

@router.post("/", response_model=ProfileResponse)
async def create_profile(
    user_id: str,
    request: UserProfile,   # <-- accept directly
    current_user_id: str = Depends(get_current_user_id)
):
    try:
        profile_response = await profile_service.create_profile(
            current_user_id,
            request   # <-- already UserProfile
        )
        return profile_response
    except Exception as e:
        logger.error(f"Error creating profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create profile")



@router.get("/{user_id}", response_model=ProfileResponse)
async def get_profile(
    user_id: str = Depends(get_current_user_id)
):
    """Get user profile by user ID"""
    try:
        profile_response = await profile_service.get_profile(user_id)
        
        if not profile_response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        logger.info(f"Profile retrieved successfully for user {user_id}")
        return profile_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )

@router.put("/{user_id}", response_model=ProfileResponse)
async def update_profile(
    request: UpdateProfileRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Update user profile"""
    try:
        # Update profile (creates if doesn't exist)
        profile_response = await profile_service.update_profile(
            user_id, 
            request.profile
        )
        
        logger.info(f"Profile updated successfully for user {user_id}")
        
        return profile_response
        
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@router.delete("/{user_id}")
async def delete_profile(
    user_id: str = Depends(get_current_user_id)
):
    """Delete user profile"""
    try:
        success = await profile_service.delete_profile(user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        logger.info(f"Profile deleted successfully for user {user_id}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Profile deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete profile"
        )

@router.get("/{user_id}/exists")
async def check_profile_exists(
    user_id: str = Depends(get_current_user_id)
):
    """Check if user profile exists"""
    try:
        exists = await profile_service.profile_exists(user_id)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"exists": exists, "user_id": user_id}
        )
        
    except Exception as e:
        logger.error(f"Error checking profile existence: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check profile existence"
        )

# Health check endpoint for profile service
@router.get("/health/check")
async def profile_health_check():
    """Health check for profile service"""
    try:
        # Test database connection
        from database.firestore import firestore_db
        db = firestore_db.get_db()
        
        # Simple test query
        test_collection = db.collection("user_profiles").limit(1)
        list(test_collection.stream())  # Execute query
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "service": "profile",
                "database": "connected"
            }
        )
        
    except Exception as e:
        logger.error(f"Profile service health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "profile",
                "error": str(e)
            }
        )