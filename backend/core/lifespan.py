import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.firebase_setup import initialize_firebase
from database.firestore import firestore_db
from database.firestore_initializer import FirestoreInitializer
from database.firestore_health import FirestoreHealth

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Student Advisor Portal...")
    
    try:
        # Initialize Firebase (Original Method)
        db = initialize_firebase()
        
        # Connect to Firestore (Enhanced Method)
        enhanced_db = firestore_db.connect()
        
        # Create instances for modular components
        firestore_initializer = FirestoreInitializer(firestore_db)
        firestore_health = FirestoreHealth(firestore_db)
        
        # Initialize collections and default data
        await firestore_initializer.create_collections_and_indexes()
        await firestore_initializer.create_default_data()
        
        logger.info("✅ Application startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Continue with fallback Firebase connection
        logger.info("🔄 Using fallback Firebase connection")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down...")
    try:
        firestore_db.disconnect()
    except Exception:
        pass
    logger.info("✅ Shutdown complete")