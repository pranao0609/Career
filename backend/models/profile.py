from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Literal
from datetime import datetime

class PersonalInfo(BaseModel):
    # Make these fields optional with defaults for incomplete profiles
    name: str = Field(default="", min_length=0, max_length=100)  # Allow empty initially
    email: EmailStr
    phone: str = Field(default="", min_length=0, max_length=15)  # Allow empty initially  
    location: str = Field(default="", min_length=0, max_length=100)  # Allow empty initially
    

class CareerInfo(BaseModel):
    current_role: str = Field(default="N/A", min_length=1, max_length=100, alias="currentRole")
    industry: str = Field(default="N/A", min_length=1, max_length=100)
    expected_salary: str = Field(default="N/A", min_length=1, max_length=50, alias="expectedSalary")
    preferred_location: str = Field(default="N/A", min_length=1, max_length=200, alias="preferredLocation")

# Enhanced Education model to match your form     # Computer Science, etc.

# Academic/Career Background model for your form data
class AcademicBackground(BaseModel):
    education_level: str = Field(..., alias="educationLevel")  # Required from form
    field_of_study: str = Field(..., alias="fieldOfStudy")     # Required from form
    years_of_experience: str = Field(..., alias="yearsOfExperience")  # Required from form
    interests: List[str] = Field(default_factory=list)         # Career interests array



# Main Profile Model with new academic background
class UserProfile(BaseModel):
    personal_info: PersonalInfo = Field(..., alias="personalInfo")
    career_info: CareerInfo = Field(..., alias="careerInfo")
    # Academic background from your form
    academic_background: Optional[AcademicBackground] = Field(None, alias="academicBackground")

# Response models remain the same
class ProfileResponse(BaseModel):
    user_id: str
    profile: UserProfile
    created_at: datetime
    updated_at: datetime

class UpdateProfileRequest(BaseModel):
    profile: UserProfile

# Specific request model for your career form
class CareerInfoFormRequest(BaseModel):
    education_level: str = Field(..., alias="educationLevel")
    field_of_study: str = Field(..., alias="fieldOfStudy")
    years_of_experience: str = Field(..., alias="yearsOfExperience")
    interests: List[str] = Field(default_factory=list)

# Response model for career form submission
class CareerInfoFormResponse(BaseModel):
    user_id: str
    academic_background: AcademicBackground
    updated_at: datetime