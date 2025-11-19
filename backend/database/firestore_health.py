import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class FirestoreHealth:
    def __init__(self, firestore_connection):
        self.firestore_connection = firestore_connection
    
    def health_check(self) -> bool:
        """Check if Firestore connection is healthy"""
        try:
            db = self.firestore_connection.get_db()
            if not db:
                return False
            
            # Try to read from Firestore
            test_ref = db.collection('health_check').document('test')
            test_ref.set({'checked_at': datetime.utcnow()})
            test_ref.delete()  # Clean up
            return True
        except Exception as e:
            logger.error(f"Firestore health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            db = self.firestore_connection.get_db()
            
            stats = {
                "database_type": "Firestore",
                "project_id": self.firestore_connection.project_id,
                "connection_status": "connected" if self.firestore_connection._connected else "disconnected",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Get collection counts (Note: This is expensive in Firestore for large collections)
            # In production, you might want to maintain counters separately
            try:
                collections = ["users", "chat_rooms", "messages", "notifications"]
                for collection in collections:
                    docs = db.collection(collection).limit(1000).get()  # Limited count for performance
                    stats[f"{collection}_count"] = len(docs) if len(docs) < 1000 else "1000+"
            except Exception as e:
                stats["count_error"] = str(e)
            
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}