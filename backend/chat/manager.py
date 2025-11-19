from typing import Dict, List, Set, Optional
from fastapi import WebSocket
import json
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """User session information"""
    user_id: str
    username: str
    websocket: WebSocket
    connected_at: datetime
    last_activity: datetime
    current_rooms: Set[str]
    is_typing_in: Optional[str] = None

class ConnectionManager:
    """Advanced WebSocket connection manager for Firestore"""
    
    def __init__(self):
        # Core connection storage
        self.active_connections: Dict[str, UserSession] = {}
        self.room_members: Dict[str, Set[str]] = {}
        self.user_typing: Dict[str, Dict[str, datetime]] = {}  # room_id -> {user_id: timestamp}
        
        # Background task management
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background cleanup tasks"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of stale connections and typing indicators"""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                await self._cleanup_stale_typing()
                await self._cleanup_inactive_sessions()
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def _cleanup_stale_typing(self):
        """Remove stale typing indicators"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=5)
        
        for room_id, typing_users in self.user_typing.items():
            stale_users = [
                user_id for user_id, timestamp in typing_users.items()
                if timestamp < cutoff_time
            ]
            
            for user_id in stale_users:
                del typing_users[user_id]
                await self._broadcast_typing_update(room_id, user_id, False)
    
    async def _cleanup_inactive_sessions(self):
        """Remove inactive user sessions"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=30)
        
        inactive_users = [
            user_id for user_id, session in self.active_connections.items()
            if session.last_activity < cutoff_time
        ]
        
        for user_id in inactive_users:
            logger.info(f"Removing inactive user: {user_id}")
            await self._force_disconnect(user_id)
    
    async def connect(self, websocket: WebSocket, user_id: str, username: str = None):
        """Accept WebSocket connection"""
        try:
            await websocket.accept()
            
            # Disconnect existing session if any
            if user_id in self.active_connections:
                await self._force_disconnect(user_id)
            
            # Create new session
            session = UserSession(
                user_id=user_id,
                username=username or user_id,
                websocket=websocket,
                connected_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                current_rooms=set()
            )
            
            self.active_connections[user_id] = session
            
            # Send welcome message
            await self._send_to_user(user_id, {
                "type": "connection_established",
                "data": {
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": str(uuid.uuid4())
                }
            })
            
            logger.info(f"✅ User {user_id} connected")
            
        except Exception as e:
            logger.error(f"Connection error for {user_id}: {e}")
            raise
    
    async def disconnect(self, user_id: str):
        """Handle user disconnection"""
        if user_id not in self.active_connections:
            return
        
        session = self.active_connections[user_id]
        
        # Remove from all rooms
        for room_id in session.current_rooms.copy():
            await self.leave_room(user_id, room_id)
        
        # Clean up typing indicators
        for room_id in self.user_typing:
            if user_id in self.user_typing[room_id]:
                del self.user_typing[room_id][user_id]
                await self._broadcast_typing_update(room_id, user_id, False)
        
        # Remove connection
        del self.active_connections[user_id]
        
        logger.info(f"❌ User {user_id} disconnected")
    
    async def _force_disconnect(self, user_id: str):
        """Force disconnect a user"""
        if user_id in self.active_connections:
            session = self.active_connections[user_id]
            try:
                await session.websocket.close(code=1000, reason="Session replaced")
            except:
                pass
            await self.disconnect(user_id)
    
    async def join_room(self, user_id: str, room_id: str):
        """Add user to a room"""
        if user_id not in self.active_connections:
            logger.warning(f"User {user_id} not connected, cannot join room {room_id}")
            return False
        
        session = self.active_connections[user_id]
        
        # Add to room
        if room_id not in self.room_members:
            self.room_members[room_id] = set()
        
        if user_id not in self.room_members[room_id]:
            self.room_members[room_id].add(user_id)
            session.current_rooms.add(room_id)
            
            # Notify room members
            await self.broadcast_to_room(room_id, {
                "type": "user_joined",
                "data": {
                    "user_id": user_id,
                    "username": session.username,
                    "room_id": room_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "member_count": len(self.room_members[room_id])
                }
            }, exclude_user=user_id)
            
            # Send room info to user
            await self._send_to_user(user_id, {
                "type": "room_joined",
                "data": {
                    "room_id": room_id,
                    "member_count": len(self.room_members[room_id]),
                    "online_members": list(self.room_members[room_id])
                }
            })
            
            logger.info(f"➡️ {user_id} joined room {room_id}")
            return True
        
        return False
    
    async def leave_room(self, user_id: str, room_id: str):
        """Remove user from a room"""
        if user_id not in self.active_connections:
            return
        
        session = self.active_connections[user_id]
        
        # Remove from room
        if room_id in self.room_members and user_id in self.room_members[room_id]:
            self.room_members[room_id].discard(user_id)
            session.current_rooms.discard(room_id)
            
            # Clean up empty rooms
            if not self.room_members[room_id]:
                del self.room_members[room_id]
            else:
                # Notify remaining members
                await self.broadcast_to_room(room_id, {
                    "type": "user_left",
                    "data": {
                        "user_id": user_id,
                        "username": session.username,
                        "room_id": room_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "member_count": len(self.room_members[room_id])
                    }
                })
            
            # Stop typing if user was typing
            if room_id in self.user_typing and user_id in self.user_typing[room_id]:
                del self.user_typing[room_id][user_id]
                await self._broadcast_typing_update(room_id, user_id, False)
            
            logger.info(f"⬅️ {user_id} left room {room_id}")
    
    async def broadcast_to_room(self, room_id: str, message: dict, exclude_user: str = None):
        """Send message to all users in a room"""
        if room_id not in self.room_members:
            return
        
        disconnected_users = []
        
        for user_id in self.room_members[room_id]:
            if exclude_user and user_id == exclude_user:
                continue
                
            if not await self._send_to_user(user_id, message):
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.disconnect(user_id)
    
    async def _send_to_user(self, user_id: str, message: dict) -> bool:
        """Send message to a specific user"""
        if user_id not in self.active_connections:
            return False
        
        session = self.active_connections[user_id]
        
        try:
            await session.websocket.send_text(json.dumps(message, default=str))
            session.last_activity = datetime.utcnow()
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to {user_id}: {e}")
            return False
    
    async def handle_typing(self, user_id: str, room_id: str, is_typing: bool):
        """Handle typing indicator"""
        if user_id not in self.active_connections:
            return
        
        if room_id not in self.user_typing:
            self.user_typing[room_id] = {}
        
        if is_typing:
            self.user_typing[room_id][user_id] = datetime.utcnow()
        elif user_id in self.user_typing[room_id]:
            del self.user_typing[room_id][user_id]
        
        await self._broadcast_typing_update(room_id, user_id, is_typing)
    
    async def _broadcast_typing_update(self, room_id: str, user_id: str, is_typing: bool):
        """Broadcast typing indicator to room members"""
        if user_id not in self.active_connections:
            return
            
        username = self.active_connections[user_id].username
        
        await self.broadcast_to_room(room_id, {
            "type": "typing_indicator",
            "data": {
                "user_id": user_id,
                "username": username,
                "room_id": room_id,
                "is_typing": is_typing,
                "timestamp": datetime.utcnow().isoformat()
            }
        }, exclude_user=user_id)
    
    # Utility methods
    def get_online_users(self) -> List[str]:
        """Get list of all online users"""
        return list(self.active_connections.keys())
    
    def get_room_members(self, room_id: str) -> List[str]:
        """Get list of users in a room"""
        return list(self.room_members.get(room_id, set()))
    
    def get_user_rooms(self, user_id: str) -> List[str]:
        """Get list of rooms a user is in"""
        if user_id in self.active_connections:
            return list(self.active_connections[user_id].current_rooms)
        return []
    
    def is_user_online(self, user_id: str) -> bool:
        """Check if user is online"""
        return user_id in self.active_connections
    
    def get_connection_stats(self) -> dict:
        """Get detailed connection statistics"""
        total_connections = len(self.active_connections)
        total_rooms_with_users = len([r for r in self.room_members.values() if r])
        
        # Calculate average session duration
        now = datetime.utcnow()
        session_durations = [
            (now - session.connected_at).total_seconds() / 60
            for session in self.active_connections.values()
        ]
        avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        return {
            "total_connections": total_connections,
            "active_rooms": total_rooms_with_users,
            "total_rooms": len(self.room_members),
            "average_session_duration_minutes": round(avg_session_duration, 2),
            "users_by_room": {
                room_id: len(members) 
                for room_id, members in self.room_members.items()
            },
            "typing_users": {
                room_id: len(typing_users)
                for room_id, typing_users in self.user_typing.items()
                if typing_users
            }
        }

# Global connection manager instance
manager = ConnectionManager()
