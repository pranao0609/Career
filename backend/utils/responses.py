from typing import Any, Dict, Optional
from fastapi import status
from fastapi.responses import JSONResponse

def success_response(
    message: str, 
    data: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_200_OK
) -> JSONResponse:
    """Create a standardized success response"""
    response_data = {
        "success": True,
        "message": message,
        "status_code": status_code
    }
    
    if data:
        response_data.update(data)
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )

def error_response(
    message: str,
    error_code: Optional[str] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create a standardized error response"""
    response_data = {
        "success": False,
        "message": message,
        "status_code": status_code
    }
    
    if error_code:
        response_data["error_code"] = error_code
    
    if details:
        response_data["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )

def paginated_response(
    data: list,
    page: int = 1,
    limit: int = 10,
    total_count: Optional[int] = None
) -> Dict[str, Any]:
    """Create a paginated response"""
    if total_count is None:
        total_count = len(data)
    
    total_pages = (total_count + limit - 1) // limit
    has_next = page < total_pages
    has_prev = page > 1
    
    return {
        "success": True,
        "data": data,
        "pagination": {
            "page": page,
            "limit": limit,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev
        }
    }

class ResponseMessages:
    """Standardized response messages"""
    
    # Authentication
    AUTH_SUCCESS = "Authentication successful"
    AUTH_FAILED = "Authentication failed"
    TOKEN_EXPIRED = "Token has expired"
    TOKEN_INVALID = "Invalid token"
    UNAUTHORIZED = "Unauthorized access"
    
    # User Management
    USER_CREATED = "User created successfully"
    USER_UPDATED = "User updated successfully"
    USER_DELETED = "User deleted successfully"
    USER_NOT_FOUND = "User not found"
    
    # Profile Management
    PROFILE_UPDATED = "Profile updated successfully"
    PROFILE_FETCH_ERROR = "Failed to fetch profile"
    
    # General
    SUCCESS = "Operation completed successfully"
    ERROR = "An error occurred"
    NOT_FOUND = "Resource not found"
    VALIDATION_ERROR = "Validation error"
    INTERNAL_ERROR = "Internal server error"
    
    # Chat
    MESSAGE_SENT = "Message sent successfully"
    MESSAGE_DELETED = "Message deleted successfully"
    ROOM_CREATED = "Chat room created successfully"
    ROOM_JOINED = "Joined chat room successfully"
    
    # Settings
    SETTINGS_UPDATED = "Settings updated successfully"
    STATUS_UPDATED = "Status updated successfully"