from pydantic import BaseModel, EmailStr
from typing import Optional

class UserSignup(BaseModel):
    """User signup model"""
    firstName: str
    lastName: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str

class GoogleAuthData(BaseModel):
    """Google authentication data"""
    name: Optional[str] = "Unknown User"
    
class TokenResponse(BaseModel):
    """Token response model"""
    message: str
    token: str
    user: dict

class AuthResponse(BaseModel):
    """Authentication response model"""
    message: str
    token: str
    uid: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    user: dict