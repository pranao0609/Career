import logging
import firebase_admin
from firebase_admin import credentials, firestore
from config.settings import get_settings

logger = logging.getLogger(__name__)

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    settings = get_settings()
    
    try:
        # Try to initialize Firebase if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            logger.info("✅ Firebase initialized successfully")
            return db
        else:
            db = firestore.client()
            logger.info("✅ Using existing Firebase connection")
            return db
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
        raise

def get_firebase_db():
    """Get Firebase Firestore client"""
    return firestore.client()