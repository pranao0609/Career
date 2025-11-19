import logging
from typing import Optional
from fastapi import HTTPException, Header, Depends
from firebase_admin import auth

from auth.jwt_handler import extract_token_from_header, decode_access_token
from database.dependencies import get_database, get_original_database

logger = logging.getLogger(__name__)

def verify_firebase_token(authorization: Optional[str] = Header(None)):
    """Verify Firebase ID token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = authorization.split(" ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        logger.error(f"Firebase token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user_from_token(authorization: Optional[str] = Header(None)):
    """Get current user from JWT token"""
    if not authorization:
        return None
    
    token = extract_token_from_header(authorization)
    if not token:
        return None
    
    payload = decode_access_token(token)
    if not payload:
        return None
    
    email = payload.get("email")
    if not email:
        return None
    
    try:
        # Try enhanced database first, then fallback to original
        try:
            database = get_database()
        except Exception:
            database = get_original_database()
        
        # Find user by email
        users = database.collection("users").where("email", "==", email).get()
        if users:
            user_data = users[0].to_dict()
            user_data["_id"] = users[0].id
            return user_data
    except Exception as e:
        logger.error(f"Token validation error: {e}")
    
    return None

def require_authentication(authorization: Optional[str] = Header(None)):
    """Require valid authentication"""
    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

def get_optional_user(authorization: Optional[str] = Header(None)):
    """Get user if authenticated, None otherwise"""
    return get_current_user_from_token(authorization)

def get_current_user(authorization: Optional[str] = Header(None)):
    """Get current user - required for career form routes"""
    return require_authentication(authorization)