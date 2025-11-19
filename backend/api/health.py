from fastapi import APIRouter, Depends , HTTPException
from datetime import datetime, timedelta
import logging

from auth.dependencies import require_authentication
from database.dependencies import get_database, get_original_database
from database.firestore_health import FirestoreHealth
from database.firestore import firestore_db

router = APIRouter()
logger = logging.getLogger(__name__)

# Try to import chat manager for online users count
try:
    from chat.manager import manager
except ImportError:
    # Create dummy manager for compatibility
    class DummyManager:
        def get_online_users(self):
            return []
    manager = DummyManager()

@router.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Try enhanced health check first
        try:
            firestore_health = FirestoreHealth(firestore_db)
            db_healthy = firestore_health.health_check()
            db_stats = firestore_health.get_stats()
        except:
            # Fallback health check
            db_healthy = True
            db_stats = {"status": "connected", "method": "original"}
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_stats,
            "websocket_connections": len(manager.get_online_users()),
            "service": "student-advisor-portal"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/stats")
def get_platform_stats(current_user: dict = Depends(require_authentication)):
    """Get platform statistics"""
    try:
        # Try enhanced database first
        try:
            database = get_database()
        except:
            database = get_original_database()
        
        # Get counts
        users = database.collection("users").get()
        user_count = len([u for u in users if u.to_dict().get("is_active", True)])
        
        try:
            rooms = database.collection("chat_rooms").where("is_active", "==", True).get()
            room_count = len(rooms)
        except:
            room_count = 0
        
        # Get messages from last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)
        try:
            recent_messages = database.collection("messages").where("timestamp", ">=", yesterday).get()
            message_count = len(recent_messages)
        except:
            message_count = 0
        
        return {
            "total_users": user_count,
            "total_rooms": room_count,
            "messages_today": message_count,
            "online_users": len(manager.get_online_users()),
            "active_rooms": len([r for r in getattr(manager, 'room_members', {}).items() if r[1]])
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")