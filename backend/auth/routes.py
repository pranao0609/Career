from fastapi import APIRouter, HTTPException, Depends, Header
from datetime import datetime
import logging
from typing import Optional
import jwt

from auth.models import UserSignup, UserLogin, GoogleAuthData
from auth.dependencies import verify_firebase_token, get_current_user_from_token
from auth.jwt_handler import create_access_token
from auth.password_handler import hash_password, verify_password
from database.dependencies import get_database, get_original_database
from config.settings import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

@router.post("/signup")
def signup(user: UserSignup):
    """User registration with enhanced features"""
    try:
        # Try enhanced database first
        try:
            database = get_database()
        except:
            database = get_original_database()
        
        # Check if user exists
        existing_users = database.collection("users").where("email", "==", user.email).get()
        if existing_users:
            raise HTTPException(status_code=400, detail="User already exists")

        # Hash password
        hashed_password = hash_password(user.password)
        
        # Create enhanced user document
        user_doc = {
            "user_id": user.email,
            "firstName": user.firstName,
            "lastName": user.lastName,
            "email": user.email,
            "password": hashed_password,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "is_active": True,
            "avatar_url": None,
            "bio": "",
            "status": "online",
            "chat_settings": {
                "notifications_enabled": True,
                "sound_enabled": True,
                "online_status_visible": True
            },
            "preferences": {
                "theme": "light",
                "language": "en"
            }
        }
        
        # Save to database
        doc_ref = database.collection("users").document()
        doc_ref.set(user_doc)
        
        # Create JWT token
        token = create_access_token({"email": user.email, "user_id": user.email})
        
        # Return user data (without password)
        user_doc.pop("password")
        user_doc["_id"] = doc_ref.id
        
        return {
            "message": "Registration successful",
            "token": token,
            "user": user_doc
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@router.post("/signin")
def signin(user: UserLogin):
    """User authentication with enhanced features"""
    try:
        # Try enhanced database first
        try:
            database = get_database()
        except:
            database = get_original_database()
        
        # Find user
        users_query = database.collection("users").where("email", "==", user.email).get()
        
        if not users_query:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_doc = users_query[0]
        user_data = user_doc.to_dict()
        
        # Verify password
        if not verify_password(user.password, user_data["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Update last activity (enhanced version)
        try:
            user_doc.reference.update({
                "last_activity": datetime.utcnow(),
                "status": "online",
                "updated_at": datetime.utcnow()
            })
        except:
            # Fallback for simpler update
            pass

        # Create JWT token
        token = create_access_token({"email": user.email, "user_id": user.email})
        
        # Return enhanced user data (without password)
        user_data.pop("password", None)
        user_data["_id"] = user_doc.id
        
        return {
            "message": "Login successful",
            "token": token,
            "user": user_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/google")
def google_auth(user_data: GoogleAuthData, decoded_token=Depends(verify_firebase_token)):
    """Google authentication"""
    try:
        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        name = user_data.name or "Unknown User"

        # Try enhanced database first
        try:
            database = get_database()
        except:
            database = get_original_database()

        # Enhanced user document for Google users
        user_ref = database.collection("users").document(uid)
        user_doc = {
            "user_id": email,
            "email": email,
            "firstName": name.split(" ")[0] if name else "Google",
            "lastName": " ".join(name.split(" ")[1:]) if len(name.split(" ")) > 1 else "User",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "is_active": True,
            "status": "online",
            "auth_provider": "google",
            "chat_settings": {
                "notifications_enabled": True,
                "sound_enabled": True,
                "online_status_visible": True
            },
            "preferences": {
                "theme": "light",
                "language": "en"
            }
        }
        
        user_ref.set(user_doc, merge=True)

        # Create JWT token for consistency
        token = create_access_token({"email": email, "user_id": email})

        return {
            "message": "Google login successful!",
            "token": token,
            "uid": uid,
            "name": name,
            "email": email,
            "user": user_doc
        }
    except Exception as e:
        logger.error(f"Google auth error: {e}")
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

@router.get("/profile")
def get_profile_simple(authorization: Optional[str] = Header(None)):
    """Get user profile (simple version for backward compatibility)"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing token")
    try:
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        email = payload.get("email")
        
        # Try enhanced database first
        try:
            database = get_database()
        except:
            database = get_original_database()
        
        user_query = database.collection("users").where("email", "==", email).get()
        if not user_query:
            raise HTTPException(status_code=404, detail="User not found")
        
        profile_data = user_query[0].to_dict()
        profile_data.pop("password", None)  # Remove password
        
        return {"profile": profile_data}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception as e:
        logger.error(f"Profile fetch error: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")