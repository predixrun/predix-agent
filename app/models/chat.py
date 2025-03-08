import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_id: str


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    message_type: str = "text"  # "text", "sport_search", "market_options", "confirmation_options", "market_finalized"
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
