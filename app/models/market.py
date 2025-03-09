from datetime import datetime
from enum import Enum

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
    name: str
    type: SelectionType
    description: str | None = None


class Market(BaseModel):
    creator_id: str
    title: str
    description: str
    type: str = "binary"
    status: MarketStatus = MarketStatus.DRAFT
    category: str = "sports"
    amount: float = 1.0
    currency: str = "SOL"
    close_date: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class EventTeam(BaseModel):
    id: int
    name: str


class EventLeague(BaseModel):
    id: int
    name: str
    country: str


class EventVenue(BaseModel):
    name: str
    city: str


class Event(BaseModel):
    type: str = "football_match"
    fixture_id: int
    home_team: EventTeam
    away_team: EventTeam
    league: EventLeague
    start_time: datetime
    venue: EventVenue | None = None


class MarketPackage(BaseModel):
    market: Market
    selections: list[Selection]
    event: Event
