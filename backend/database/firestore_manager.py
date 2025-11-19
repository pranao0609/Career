from .firestore import firestore_db
from .firestore_initializer import FirestoreInitializer
from .firestore_health import FirestoreHealth

class FirestoreManager:
    def __init__(self):
        self.connection = firestore_db
        self.initializer = FirestoreInitializer(self.connection)
        self.health = FirestoreHealth(self.connection)
    
    def get_db(self):
        """Get database connection"""
        return self.connection.get_db()
    
    def connect(self):
        """Connect to database"""
        return self.connection.connect()
    
    def disconnect(self):
        """Disconnect from database"""
        self.connection.disconnect()
    
    async def initialize_database(self):
        """Initialize database with collections and default data"""
        await self.initializer.create_collections_and_indexes()
        await self.initializer.create_default_data()
    
    def health_check(self):
        """Check database health"""
        return self.health.health_check()
    
    def get_stats(self):
        """Get database statistics"""
        return self.health.get_stats()
    
    def run_async(self, coro):
        """Helper to run async operations"""
        return self.connection.run_async(coro)

# Global manager instance
firestore_manager = FirestoreManager()