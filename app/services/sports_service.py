import random
from datetime import datetime, timedelta
from typing import Any

# import httpx
# from app.config import logger, settings
# from app.models.sports import ApiResponse, FixtureResponse, TeamDetail


async def search_teams(query: str) -> list[dict[str, Any]]:
    """
    Search for football teams by name.
    This is a mock implementation for PoC.

    Args:
        query: Team name to search for

    Returns:
        List of matching teams
    """
    # Mock response for development
    mock_teams = [
        {
            "team": {
                "id": 33,
                "name": "Manchester United",
                "code": "MUN",
                "country": "England",
                "founded": 1878,
                "national": False,
                "logo": "https://media.api-sports.io/football/teams/33.png"
            },
            "venue": {
                "id": 556,
                "name": "Old Trafford",
                "address": "Sir Matt Busby Way",
                "city": "Manchester",
                "capacity": 76212,
                "surface": "grass",
                "image": "https://media.api-sports.io/football/venues/556.png"
            }
        },
        {
            "team": {
                "id": 40,
                "name": "Manchester City",
                "code": "MCI",
                "country": "England",
                "founded": 1880,
                "national": False,
                "logo": "https://media.api-sports.io/football/teams/40.png"
            },
            "venue": {
                "id": 555,
                "name": "Etihad Stadium",
                "address": "Rowsley Street",
                "city": "Manchester",
                "capacity": 55097,
                "surface": "grass",
                "image": "https://media.api-sports.io/football/venues/555.png"
            }
        }
    ]

    # todo:: 실제로 구현, real API call:
    # headers = {
    #     'x-rapidapi-host': "v3.football.api-sports.io",
    #     'x-rapidapi-key': settings.FOOTBALL_API_KEY
    # }
    #
    # async with httpx.AsyncClient() as client:
    #     response = await client.get(
    #         f"https://v3.football.api-sports.io/teams?search={query}",
    #         headers=headers
    #     )
    #     if response.status_code == 200:
    #         data = response.json()
    #         return data.get("response", [])

    # Filter mock data based on query
    query = query.lower()
    return [team for team in mock_teams if query in team["team"]["name"].lower()]


async def get_fixtures(team_id: int | None = None,
                       league_id: int | None = None,
                       date: str | None = None,
                       upcoming: bool = True) -> list[dict[str, Any]]:
    """
    Get football fixtures by team, league, or date.
    Mock implementation for PoC.

    Args:
        team_id: Optional team ID to filter by
        league_id: Optional league ID to filter by
        date: Optional date to filter by (YYYY-MM-DD)
        upcoming: If True, get upcoming fixtures; if False, get past fixtures

    Returns:
        List of fixtures
    """
    # Mock response for development
    now = datetime.now()

    if upcoming:
        # Generate fixtures for next 7 days
        fixtures = []
        for i in range(1, 8):
            fixture_date = now + timedelta(days=i)
            fixtures.append({
                "fixture": {
                    "id": 1000 + i,
                    "referee": "M. Oliver",
                    "timezone": "UTC",
                    "date": fixture_date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "timestamp": int(fixture_date.timestamp()),
                    "status": {
                        "long": "Not Started",
                        "short": "NS"
                    }
                },
                "league": {
                    "id": 39,
                    "name": "Premier League",
                    "country": "England",
                    "logo": "https://media.api-sports.io/football/leagues/39.png",
                    "flag": "https://media.api-sports.io/flags/gb.svg",
                    "season": 2024,
                    "round": f"Regular Season - {random.randint(1, 38)}"
                },
                "teams": {
                    "home": {
                        "id": 33 if i % 2 == 0 else 40,
                        "name": "Manchester United" if i % 2 == 0 else "Manchester City",
                        "logo": f"https://media.api-sports.io/football/teams/{33 if i % 2 == 0 else 40}.png"
                    },
                    "away": {
                        "id": 40 if i % 2 == 0 else 33,
                        "name": "Manchester City" if i % 2 == 0 else "Manchester United",
                        "logo": f"https://media.api-sports.io/football/teams/{40 if i % 2 == 0 else 33}.png"
                    }
                },
                "goals": {
                    "home": None,
                    "away": None
                },
                "score": {
                    "halftime": {
                        "home": None,
                        "away": None
                    },
                    "fulltime": {
                        "home": None,
                        "away": None
                    },
                    "extratime": {
                        "home": None,
                        "away": None
                    },
                    "penalty": {
                        "home": None,
                        "away": None
                    }
                }
            })
    else:
        # Generate past fixtures
        fixtures = []
        for i in range(1, 8):
            fixture_date = now - timedelta(days=i)
            fixtures.append({
                "fixture": {
                    "id": 900 + i,
                    "referee": "M. Oliver",
                    "timezone": "UTC",
                    "date": fixture_date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "timestamp": int(fixture_date.timestamp()),
                    "status": {
                        "long": "Match Finished",
                        "short": "FT"
                    }
                },
                "league": {
                    "id": 39,
                    "name": "Premier League",
                    "country": "England",
                    "logo": "https://media.api-sports.io/football/leagues/39.png",
                    "flag": "https://media.api-sports.io/flags/gb.svg",
                    "season": 2024,
                    "round": f"Regular Season - {random.randint(1, 38)}"
                },
                "teams": {
                    "home": {
                        "id": 33 if i % 2 == 0 else 40,
                        "name": "Manchester United" if i % 2 == 0 else "Manchester City",
                        "logo": f"https://media.api-sports.io/football/teams/{33 if i % 2 == 0 else 40}.png"
                    },
                    "away": {
                        "id": 40 if i % 2 == 0 else 33,
                        "name": "Manchester City" if i % 2 == 0 else "Manchester United",
                        "logo": f"https://media.api-sports.io/football/teams/{40 if i % 2 == 0 else 33}.png"
                    }
                },
                "goals": {
                    "home": random.randint(0, 4),
                    "away": random.randint(0, 3)
                },
                "score": {
                    "halftime": {
                        "home": random.randint(0, 2),
                        "away": random.randint(0, 2)
                    },
                    "fulltime": {
                        "home": random.randint(0, 4),
                        "away": random.randint(0, 3)
                    },
                    "extratime": {
                        "home": None,
                        "away": None
                    },
                    "penalty": {
                        "home": None,
                        "away": None
                    }
                }
            })

    # Filter by team_id if provided
    if team_id:
        fixtures = [
            f for f in fixtures if
            f["teams"]["home"]["id"] == team_id or
            f["teams"]["away"]["id"] == team_id
        ]

    # Filter by league_id if provided
    if league_id:
        fixtures = [f for f in fixtures if f["league"]["id"] == league_id]

    # Filter by date if provided
    if date:
        fixtures = [
            f for f in fixtures if
            f["fixture"]["date"].startswith(date)
        ]

    return fixtures
