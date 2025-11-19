from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Enums
class MessageType(str, Enum):
    TEXT = "text"
    FILE = "file"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SYSTEM = "system"

class RoomType(str, Enum):
    GENERAL = "general"
    COURSE = "course"
    STUDY_GROUP = "study_group"
    PROJECT = "project"
    HELP = "help"
    PRIVATE = "private"

class UserStatus(str, Enum):
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"

# Core Models
class ChatUser(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    username: str
    email: str
    avatar_url: Optional[str] = None
    status: UserStatus = UserStatus.ONLINE
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_online: bool = False
    bio: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = {}
    
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

class RoomSettings(BaseModel):
    allow_file_upload: bool = True
    allow_reactions: bool = True
    max_file_size_mb: int = 10
    moderated: bool = False
    require_approval: bool = False
    mute_notifications: bool = False

class ChatRoom(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    room_id: str
    name: str
    description: Optional[str] = ""
    room_type: RoomType = RoomType.GENERAL
    members: List[str] = []
    moderators: List[str] = []
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    is_public: bool = True
    max_members: Optional[int] = None
    settings: RoomSettings = Field(default_factory=RoomSettings)
    tags: List[str] = []
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Room name cannot be empty')
        return v.strip()
    
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

class MessageReaction(BaseModel):
    emoji: str
    users: List[str] = []
    count: int = 0

class MessageAttachment(BaseModel):
    filename: str
    url: str
    size: int
    mime_type: str
    thumbnail_url: Optional[str] = None

class Message(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    message_id: str
    room_id: str
    sender_id: str
    sender_name: str
    content: str
    message_type: MessageType = MessageType.TEXT
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    edited_at: Optional[datetime] = None
    is_edited: bool = False
    is_deleted: bool = False
    is_pinned: bool = False
    reply_to: Optional[str] = None  # message_id
    thread_id: Optional[str] = None
    attachments: List[MessageAttachment] = []
    reactions: List[MessageReaction] = []
    mentions: List[str] = []  # user_ids
    
    @validator('content')
    def content_must_not_be_empty_for_text(cls, v, values):
        if values.get('message_type') == MessageType.TEXT and (not v or not v.strip()):
            raise ValueError('Text message content cannot be empty')
        return v
    
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}

# Request/Response Models
class RoomCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    room_type: RoomType = RoomType.GENERAL
    is_public: bool = True
    max_members: Optional[int] = None
    settings: Optional[RoomSettings] = None
    tags: List[str] = []
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Room name must be at least 2 characters')
        if len(v.strip()) > 100:
            raise ValueError('Room name cannot exceed 100 characters')
        return v.strip()

class MessageCreateRequest(BaseModel):
    content: str
    message_type: MessageType = MessageType.TEXT
    reply_to: Optional[str] = None
    mentions: List[str] = []
    
    @validator('content')
    def content_length_check(cls, v):
        if len(v) > 2000:
            raise ValueError('Message cannot exceed 2000 characters')
        return v

class MessageEditRequest(BaseModel):
    content: str
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message content cannot be empty')
        return v.strip()

class RoomUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None
    max_members: Optional[int] = None
    settings: Optional[RoomSettings] = None
    tags: Optional[List[str]] = None

# WebSocket Models
class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TypingIndicator(BaseModel):
    room_id: str
    user_id: str
    is_typing: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Response Models
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None

class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool

class ConnectionStats(BaseModel):
    total_connections: int
    active_rooms: int
    messages_today: int
    peak_concurrent_users: int
    average_session_duration: float
