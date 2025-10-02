import logging
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytz
from dateutil import parser

from app.config import settings

# API Configuration
API_BASE_URL = "https://v3.football.api-sports.io"
API_HEADERS = {
    'x-apisports-key': settings.SPORTS_API_KEY
}


async def search_teams(query: str) -> list[dict[str, Any]]:
    """
    Search for football teams by name using the api-sports.io API.

    Args:
        query: Team name to search for

    Returns:
        List of matching teams
    """
    # logging.info(f"Searching for teams with query: {query}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/teams",
                headers=API_HEADERS,
                params={"search": query}
            )

            if response.status_code == 200:
                data = response.json()
                # logging.info(f"Found {data.get('results', 0)} teams matching '{query}'")
                return data.get("response", [])
            else:
                logging.error(f"API error searching teams: {response.status_code} - {response.text}")
                return []
    except Exception as e:
        logging.error(f"Error searching teams: {e}")
        return []


async def get_fixtures(
        team_id: int | None = None,
        league_id: int | None = None,
        date: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        fixture_id: int | None = None,
        upcoming: bool = True
) -> list[dict[str, Any]]:
    """
    Get football fixtures by team, league, date, or fixture ID using the api-sports.io API.

    Args:
        team_id: Optional team ID to filter by
        league_id: Optional league ID to filter by
        date: Optional specific date to filter by (YYYY-MM-DD)
        from_date: Optional start date for date range (YYYY-MM-DD)
        to_date: Optional end date for date range (YYYY-MM-DD)
        fixture_id: Optional specific fixture ID to retrieve
        upcoming: If True, get upcoming fixtures; if False, get past fixtures

    Returns:
        List of fixtures
    """
    # Prepare parameters
    params = {"timezone": "UTC", "season": 2025}  # Use UTC timezone

    if fixture_id:
        # logging.info(f"Getting fixture with ID: {fixture_id}")
        params["id"] = fixture_id
    else:
        # For upcoming/past fixtures when no specific date is provided
        if not date and not from_date and not to_date:
            # Calculate date range
            today = datetime.now().date()

            if upcoming:
                # For upcoming matches, use next 7 days
                params["from"] = today.strftime("%Y-%m-%d")
                params["to"] = (today + timedelta(days=7)).strftime("%Y-%m-%d")
                # logging.info(f"Getting upcoming fixtures from {params['from']} to {params['to']}")
            else:
                # For past matches, use last 7 days
                params["from"] = (today - timedelta(days=7)).strftime("%Y-%m-%d")
                params["to"] = today.strftime("%Y-%m-%d")
                # logging.info(f"Getting past fixtures from {params['from']} to {params['to']}")

        # Add specific date if provided
        if date:
            # logging.info(f"Getting fixtures for date: {date}")
            params["date"] = date

        # Add date range if provided
        if from_date:
            # logging.info(f"Getting fixtures from date: {from_date}")
            params["from"] = from_date
        if to_date:
            # logging.info(f"Getting fixtures to date: {to_date}")
            params["to"] = to_date

        # Add team ID if provided
        if team_id:
            # logging.info(f"Getting fixtures for team ID: {team_id}")
            params["team"] = team_id

        # Add league ID if provided
        if league_id:
            # logging.info(f"Getting fixtures for league ID: {league_id}")
            params["league"] = league_id

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/fixtures",
                headers=API_HEADERS,
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                fixtures = data.get("response", [])
                # logging.info(f"Found {len(fixtures)} fixtures matching criteria")

                # Return filtered results based on upcoming flag if not using fixture_id
                if not fixture_id and fixtures:
                    now = datetime.now(pytz.UTC)

                    if upcoming:
                        # Filter for upcoming fixtures
                        return [
                            fixture for fixture in fixtures
                            if parser.parse(fixture["fixture"]["date"]) > now
                        ]
                    else:
                        # Filter for past fixtures
                        return [
                            fixture for fixture in fixtures
                            if parser.parse(fixture["fixture"]["date"]) <= now
                        ]

                return fixtures
            else:
                logging.error(f"API error getting fixtures: {response.status_code} - {response.text}")
                return []
    except Exception as e:
        logging.error(f"Error getting fixtures: {e}")
        return []


async def get_leagues(
        country: str | None = None,
        search: str | None = None,
        league_id: int | None = None
) -> list[dict[str, Any]]:
    """
    Get football leagues information.

    Args:
        country: Optional country name to filter leagues
        search: Optional search term to filter leagues by name
        league_id: Optional specific league ID to retrieve

    Returns:
        List of leagues
    """
    # logging.info(f"Getting leagues with filters - country: {country}, search: {search}, id: {league_id}")

    params = {}
    if country:
        params["country"] = country
    if search:
        params["search"] = search
    if league_id:
        params["id"] = league_id

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/leagues",
                headers=API_HEADERS,
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                leagues = data.get("response", [])
                # logging.info(f"Found {len(leagues)} leagues matching criteria")
                return leagues
            else:
                logging.error(f"API error getting leagues: {response.status_code} - {response.text}")
                return []
    except Exception as e:
        logging.error(f"Error getting leagues: {e}")
        return []


async def get_fixture_details(fixture_id: int) -> dict[str, Any]:
    """
    특정 fixture ID로 경기 상세 정보를 조회합니다.

    Args:
        fixture_id: 조회할 경기 ID

    Returns:
        경기 상세 정보 또는 찾을 수 없는 경우 빈 딕셔너리
    """
    logging.info(f"Getting fixture details for ID: {fixture_id}")
    fixtures = await get_fixtures(fixture_id=fixture_id)

    if not fixtures or len(fixtures) == 0:
        logging.error(f"No fixture found with ID: {fixture_id}")
        return {}

    # 첫 번째(유일한) 경기 정보 반환
    return fixtures[0]


def preprocess_sports_data(sports_data: dict[str, list]) -> dict[str, list]:
    """
    Preprocess and limit sports data to make it more manageable for the LLM.

    Args:
        sports_data: Original sports data with fixtures and teams

    Returns:
        Preprocessed and limited sports data
    """
    processed_data = {
        "fixtures": [],
        "teams": []
    }

    # Process teams data (limit to top 5)
    if "teams" in sports_data and sports_data["teams"]:
        processed_data["teams"] = sports_data["teams"][:5]

    # Process fixtures data (limit to top 10)
    if "fixtures" in sports_data and sports_data["fixtures"]:
        fixtures = sports_data["fixtures"][:10]  # Limit to top 10 fixtures

        for fixture in fixtures:
            # Simplify and clean up fixture data
            processed_fixture = {
                "fixture": {
                    "id": fixture["fixture"]["id"],
                    "date": fixture["fixture"]["date"],
                    "status": fixture["fixture"]["status"]["long"],
                    "venue": fixture["fixture"].get("venue", {}).get("name", "Unknown Venue")
                },
                "league": {
                    "name": fixture["league"]["name"],
                    "country": fixture["league"]["country"],
                    "round": fixture["league"].get("round", "")
                },
                "teams": {
                    "home": {
                        "id": fixture["teams"]["home"]["id"],
                        "name": fixture["teams"]["home"]["name"],
                        "winner": fixture["teams"]["home"].get("winner")
                    },
                    "away": {
                        "id": fixture["teams"]["away"]["id"],
                        "name": fixture["teams"]["away"]["name"],
                        "winner": fixture["teams"]["away"].get("winner")
                    }
                },
                "score": {
                    "home": fixture["goals"]["home"],
                    "away": fixture["goals"]["away"]
                }
            }

            processed_data["fixtures"].append(processed_fixture)

    return processed_data
