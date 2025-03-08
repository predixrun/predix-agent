import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MarketStatus(str, Enum):
    """Enum for market status"""
    DRAFT = "draft"
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    CANCELLED = "cancelled"


class SelectionType(str, Enum):
    """Enum for selection type"""
    WIN = "win"
    DRAW_LOSE = "draw_lose"


class Selection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str | None = None
    type: SelectionType


class Market(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    creator_id: str
    title: str
    description: str
    type: str = "binary"
    status: MarketStatus = MarketStatus.DRAFT
    selections: list[Selection]
    close_date: datetime | None = None
    category: str = "sports"
    event_details: dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class UserBet(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    market_id: str
    selection_id: str
    amount: float
    currency: str = "SOL"
    timestamp: datetime = Field(default_factory=datetime.now)


class SelectionRequest(BaseModel):
    user_id: str
    conversation_id: str
    market_id: str
    selection_id: str
    amount: float = 1.0


class ConfirmationRequest(BaseModel):
    user_id: str
    conversation_id: str
    market_id: str
    confirmed: bool

