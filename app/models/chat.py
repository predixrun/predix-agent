import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Enum for message types in chat responses"""
    TEXT = "TEXT"
    SPORTS_SEARCH = "SPORTS_SEARCH"
    MARKET_OPTIONS = "MARKET_OPTIONS"
    MARKET_FINALIZED = "MARKET_FINALIZED"
    ERROR = "ERROR"
    TOKEN_SWAP = "TOKEN_SWAP"


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

    def to_langchain_message(self) -> BaseMessage:
        """Convert to LangChain message format"""
        if self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role == "assistant":
            return AIMessage(content=self.content)
        elif self.role == "system":
            return SystemMessage(content=self.content)
        else:
            # Default fallback to HumanMessage
            return HumanMessage(content=self.content)

    @classmethod
    def from_langchain_message(cls, message: BaseMessage) -> "Message":
        """Create from LangChain message."""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "user"

        return cls(role=role, content=message.content)


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    message: list[Message] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
