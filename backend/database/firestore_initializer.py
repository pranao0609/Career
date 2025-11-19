import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class FirestoreInitializer:
    def __init__(self, firestore_connection):
        self.firestore_connection = firestore_connection
    
    async def create_collections_and_indexes(self):
        """Create default collections and sample data"""
        try:
            db = self.firestore_connection.get_db()
            
            # Firestore creates collections automatically when you write data
            # We'll create some sample documents to initialize collections
            
            logger.info("📁 Initializing Firestore collections...")
            
            # Initialize collections by creating sample documents
            collections_init = [
                ('users', {'_type': 'collection_init', 'created_at': datetime.utcnow()}),
                ('chat_rooms', {'_type': 'collection_init', 'created_at': datetime.utcnow()}),
                ('messages', {'_type': 'collection_init', 'created_at': datetime.utcnow()}),
                ('user_sessions', {'_type': 'collection_init', 'created_at': datetime.utcnow()}),
                ('notifications', {'_type': 'collection_init', 'created_at': datetime.utcnow()})
            ]
            
            for collection_name, init_doc in collections_init:
                # Check if collection has any documents
                docs = db.collection(collection_name).limit(1).get()
                if not docs:
                    # Collection is empty, add initialization document
                    doc_ref = db.collection(collection_name).document('_init')
                    doc_ref.set(init_doc)
                    logger.info(f"✅ Initialized collection: {collection_name}")
                    
                    # Delete the init document after a moment
                    doc_ref.delete()
            
            logger.info("✅ Firestore collections initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing collections: {e}")
    
    async def create_default_data(self):
        """Create default rooms and sample data"""
        try:
            db = self.firestore_connection.get_db()
            
            # Default chat rooms
            default_rooms = self._get_default_rooms()
            
            # Create rooms if they don't exist
            for room_data in default_rooms:
                # Check if room exists
                existing_rooms = db.collection("chat_rooms").where("room_id", "==", room_data["room_id"]).get()
                
                if not existing_rooms:
                    # Create new room
                    doc_ref = db.collection("chat_rooms").document()
                    doc_ref.set(room_data)
                    logger.info(f"✅ Created room: {room_data['name']}")
                else:
                    logger.info(f"📁 Room already exists: {room_data['name']}")
                    
        except Exception as e:
            logger.error(f"❌ Error creating default data: {e}")
    
    def _get_default_rooms(self) -> List[Dict[str, Any]]:
        """Get default chat rooms configuration"""
        return [
            {
                "room_id": "general",
                "name": "General Discussion",
                "description": "Welcome! Chat about anything here 💬",
                "room_type": "general",
                "members": [],
                "moderators": ["system"],
                "created_by": "system",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "is_active": True,
                "is_public": True,
                "max_members": None,
                "settings": {
                    "allow_file_upload": True,
                    "allow_reactions": True,
                    "moderated": False,
                    "require_approval": False
                },
                "tags": ["general", "chat"]
            },
            {
                "room_id": "study-help",
                "name": "Study Help & Support", 
                "description": "Get help with your studies and coursework 📚",
                "room_type": "help",
                "members": [],
                "moderators": ["system"],
                "created_by": "system",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "is_active": True,
                "is_public": True,
                "max_members": None,
                "settings": {
                    "allow_file_upload": True,
                    "allow_reactions": True,
                    "moderated": True,
                    "require_approval": False
                },
                "tags": ["study", "help", "academic"]
            },
            {
                "room_id": "project-collaboration",
                "name": "Project Collaboration",
                "description": "Work together on exciting projects 🚀",
                "room_type": "project",
                "members": [],
                "moderators": ["system"],
                "created_by": "system",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "is_active": True,
                "is_public": True,
                "max_members": 50,
                "settings": {
                    "allow_file_upload": True,
                    "allow_reactions": True,
                    "moderated": False,
                    "require_approval": False
                },
                "tags": ["projects", "collaboration", "development"]
            },
            {
                "room_id": "random-chat",
                "name": "Random Chat",
                "description": "Casual conversations and fun topics 🎉",
                "room_type": "general",
                "members": [],
                "moderators": ["system"],
                "created_by": "system",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "is_active": True,
                "is_public": True,
                "max_members": 100,
                "settings": {
                    "allow_file_upload": False,
                    "allow_reactions": True,
                    "moderated": False,
                    "require_approval": False
                },
                "tags": ["random", "fun", "casual"]
            }
        ]