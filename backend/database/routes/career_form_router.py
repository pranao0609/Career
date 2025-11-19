from fastapi import APIRouter, HTTPException, Path
from typing import Dict, Any
import logging
from datetime import datetime
from pydantic import BaseModel

from ..profile_service import profile_service
from models.profile import UserProfile, CareerInfo, AcademicBackground, PersonalInfo

router = APIRouter(prefix="/api/career-form")
logger = logging.getLogger(__name__)

class CareerFormPayload(BaseModel):
    careerInfo: Dict[str, Any]

@router.post("/academic-background/{user_email}")
async def update_career_info(
    user_email: str = Path(..., description="User email as identifier"),
    payload: CareerFormPayload = None
):
    """
    Update user's career information - simplified version for clean data model
    """
    try:
        logger.info(f"Updating career info for user: {user_email}")
        
        if not payload or not payload.careerInfo:
            raise HTTPException(status_code=400, detail="Invalid payload")
        
        data = payload.careerInfo
        
        # Extract required fields
        education_level = data.get("educationLevel")
        field_of_study = data.get("fieldOfStudy")
        years_of_experience = data.get("yearsOfExperience")
        interests = data.get("interests", [])
        current_role = data.get("currentRole")
        industry = data.get("industry")
        expected_salary = data.get("expectedSalary")
        preferred_location = data.get("preferredLocation")
        
        # Validate required fields
        required_fields = [education_level, field_of_study, years_of_experience, current_role, industry, expected_salary, preferred_location]
        if not all(required_fields):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Get existing profile
        existing_profile = await profile_service.get_profile(user_email)
        
        if existing_profile:
            # Update existing profile
            profile = existing_profile.profile
            
            # Update career info
            profile.career_info = CareerInfo(
                currentRole=current_role,
                industry=industry,
                expectedSalary=expected_salary,
                preferredLocation=preferred_location
            )
            
            # Update academic background
            profile.academic_background = AcademicBackground(
                educationLevel=education_level,
                fieldOfStudy=field_of_study,
                yearsOfExperience=years_of_experience,
                interests=interests
            )
            
            result = await profile_service.update_profile(user_email, profile)
            
        else:
            # Create minimal profile for career-first users
            minimal_personal = PersonalInfo(
                name="",
                email=user_email,
                phone="",
                location=""
            )
            
            career_info = CareerInfo(
                currentRole=current_role,
                industry=industry,
                expectedSalary=expected_salary,
                preferredLocation=preferred_location
            )
            
            academic_background = AcademicBackground(
                educationLevel=education_level,
                fieldOfStudy=field_of_study,
                yearsOfExperience=years_of_experience,
                interests=interests
            )
            
            new_profile = UserProfile(
                personalInfo=minimal_personal,
                careerInfo=career_info,
                academicBackground=academic_background
            )
            
            result = await profile_service.create_or_update_profile(user_email, new_profile)
        
        return {
            "success": True,
            "message": "Career information updated successfully",
            "user_id": user_email,
            "updated_at": result.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating career info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/academic-background/{user_email}")
async def get_career_info(user_email: str):
    """Get user's career information"""
    try:
        profile_response = await profile_service.get_profile(user_email)
        
        if not profile_response:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return {
            "success": True,
            "data": profile_response.profile.model_dump(by_alias=True)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting career info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))