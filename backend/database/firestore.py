import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

class FirestoreConnection:
    def __init__(self):
        self.db: Optional[firestore.Client] = None
        self.app: Optional[firebase_admin.App] = None
        self._connected = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Configuration
        self.project_id = os.getenv("FIREBASE_PROJECT_ID", "ai-advisor-86f45")
        self.credentials_path = os.getenv("FIREBASE_CREDENTIALS", "serviceAccountKey.json")
    
    def connect(self) -> firestore.Client:
        """Connect to Firestore database"""
        if self._connected and self.db:
            return self.db
        
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # Ensure credentials file exists
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"Firebase credentials not found at: {self.credentials_path}")
                
                # Initialize Firebase Admin SDK
                cred = credentials.Certificate(self.credentials_path)
                self.app = firebase_admin.initialize_app(cred, {
                    'projectId': self.project_id
                })
                logger.info("🔥 Firebase Admin SDK initialized")
            else:
                self.app = firebase_admin.get_app()
            
            # Get Firestore client
            self.db = firestore.client()
            self._connected = True
            
            # Test connection
            self._test_connection()
            
            logger.info("✅ Firestore connected successfully")
            return self.db
            
        except Exception as e:
            logger.error(f"❌ Firestore connection failed: {e}")
            raise Exception(f"Failed to connect to Firestore: {str(e)}")
    
    def _test_connection(self):
        """Test the Firestore connection"""
        try:
            # Try to read from firestore (this will work even if collection doesn't exist)
            test_ref = self.db.collection('connection_test').document('test')
            test_ref.set({'connected_at': datetime.utcnow()})
            logger.info("🔗 Firestore connection test successful")
            
            # Clean up test document
            test_ref.delete()
        except Exception as e:
            raise Exception(f"Firestore connection test failed: {str(e)}")
    
    def get_db(self) -> firestore.Client:
        """Get Firestore database instance"""
        if not self._connected:
            return self.connect()
        return self.db
    
    def run_async(self, coro):
        """Helper to run async operations in sync context"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is already running, use executor
                future = asyncio.ensure_future(coro)
                return future
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop in current thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
    
    def disconnect(self):
        """Disconnect from Firestore (cleanup)"""
        try:
            if self.app:
                firebase_admin.delete_app(self.app)
            self._connected = False
            self.db = None
            self.app = None
            self.executor.shutdown(wait=True)
            logger.info("🔥 Firebase Admin SDK disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting from Firebase: {e}")

# Global Firestore instance
firestore_db = FirestoreConnection()