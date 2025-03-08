from datetime import datetime
from typing import Any

from pydantic import BaseModel


class Team(BaseModel):
    id: int
    name: str
    code: str | None = None
    country: str | None = None
    logo: str | None = None


class Venue(BaseModel):
    id: int | None = None
    name: str | None = None
    city: str | None = None
    capacity: int | None = None


class TeamDetail(BaseModel):
    team: Team
    venue: Venue | None = None


class League(BaseModel):
    id: int
    name: str
    country: str
    logo: str | None = None
    season: int


class FixtureStatus(BaseModel):
    long: str
    short: str


class FixtureTeams(BaseModel):
    home: Team
    away: Team


class FixtureScore(BaseModel):
    home: int | None = None
    away: int | None = None


class FixtureScores(BaseModel):
    halftime: FixtureScore | None = None
    fulltime: FixtureScore | None = None
    extratime: FixtureScore | None = None
    penalty: FixtureScore | None = None


class Fixture(BaseModel):
    id: int
    referee: str | None = None
    timezone: str
    date: datetime
    venue: Venue | None = None
    status: FixtureStatus


class FixtureResponse(BaseModel):
    fixture: Fixture
    league: League
    teams: FixtureTeams
    goals: FixtureScore
    score: FixtureScores


class ApiResponse(BaseModel):
    results: int
    response: list[Any]
