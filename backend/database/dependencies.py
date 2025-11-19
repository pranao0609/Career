from database.firestore import firestore_db
from core.firebase_setup import get_firebase_db

def get_database():
    """Get Firestore database instance (Enhanced Method)"""
    try:
        return firestore_db.get_db()
    except Exception:
        # Fallback to original Firebase connection
        return get_firebase_db()

def get_original_database():
    """Get original Firestore database instance"""
    return get_firebase_db()