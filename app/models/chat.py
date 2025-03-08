import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Enum for message types in chat responses"""
    TEXT = "text"
    SPORTS_SEARCH = "sports_search"
    MARKET_OPTIONS = "market_options"
    BETTING_AMOUNT_REQUEST = "betting_amount_request"
    MARKET_FINALIZED = "market_finalized"
    ERROR = "error"


class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_id: str


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    message_type: MessageType = MessageType.TEXT
    data: dict[str, Any] | None = None


class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    message: list[Message] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
