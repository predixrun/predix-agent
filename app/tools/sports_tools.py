import logging

from langchain.tools import StructuredTool

from app.services.sports_service import get_fixtures, get_leagues, preprocess_sports_data, search_teams


async def league_search(
        search: str | None = None,
        country: str | None = None
) -> dict:
    """
    리그 검색 도구

    Args:
        search: 리그 이름 검색어
        country: 국가 이름

    Returns:
        검색 결과와 스포츠 데이터
    """
    # logging.info(f"Searching leagues - search: '{search}', country: '{country}'")

    try:
        leagues = await get_leagues(search=search, country=country)

        if not leagues:
            return {
                "message": f"No leagues found for '{search}' in {country or 'any country'}.",
                "leagues": [],
                "sports_data": {}
            }

        processed_data = {"leagues": leagues}

        return {
            "message": f"Found {len(leagues)} leagues matching your criteria.",
            "leagues": [
                {
                    "id": league["league"]["id"],
                    "name": league["league"]["name"],
                    "country": league["country"]["name"],
                    "season": league["seasons"][0]["year"] if league.get("seasons") else None
                }
                for league in leagues[:5]  # Limit to top 5 leagues
            ],
            "sports_data": preprocess_sports_data(processed_data)
        }

    except Exception as e:
        logging.error(f"Error searching leagues: {str(e)}")
        return {
            "message": "Error searching leagues. Please try again.",
            "leagues": [],
            "sports_data": {}
        }

async def team_search(
        name: str,
        league_id: int | None = None
) -> dict:
    """
    팀 검색 도구

    Args:
        name: 팀 이름
        league_id: 리그 ID (선택 사항)

    Returns:
        검색 결과와 스포츠 데이터
    """
    # logging.info(f"Searching teams - name: '{name}', league_id: {league_id}")

    try:
        teams = await search_teams(name)

        if not teams:
            return {
                "message": f"No teams found for '{name}'.",
                "teams": [],
                "sports_data": {}
            }

        # Filter by league_id if provided
        if league_id:
            teams = [team for team in teams if team.get("team", {}).get("league_id") == league_id]

        # Add team fixtures
        fixtures = []
        for team in teams[:3]:  # Limit to top 3 teams to avoid too many API calls
            team_id = team["team"]["id"]
            team_fixtures = await get_fixtures(team_id=team_id, upcoming=True)
            fixtures.extend(team_fixtures)

        processed_data = {
            "teams": teams,
            "fixtures": fixtures
        }

        return {
            "message": f"Found {len(teams)} teams matching '{name}'. Showing upcoming fixtures.",
            "teams": [
                {
                    "id": team["team"]["id"],
                    "name": team["team"]["name"],
                    "country": team.get("team", {}).get("country", "Unknown"),
                    "logo": team.get("team", {}).get("logo")
                }
                for team in teams[:5]  # Limit to top 5 teams
            ],
            "fixtures": [
                {
                    "id": fixture["fixture"]["id"],
                    "date": fixture["fixture"]["date"],
                    "home_team": fixture["teams"]["home"]["name"],
                    "away_team": fixture["teams"]["away"]["name"],
                    "league": fixture["league"]["name"]
                }
                for fixture in fixtures[:5]  # Limit to top 5 fixtures
            ],
            "sports_data": preprocess_sports_data(processed_data)
        }

    except Exception as e:
        logging.error(f"Error searching teams: {str(e)}")
        return {
            "message": "Error searching teams. Please try again.",
            "teams": [],
            "fixtures": [],
            "sports_data": {}
        }

async def fixture_search(
        date: str | None = None,
        team_id: int | None = None,
        league_id: int | None = None,
        fixture_id: int | None = None,
        upcoming: bool = True
) -> dict:
    """
    경기 검색 도구

    Args:
        date: 날짜 (YYYY-MM-DD)
        team_id: 팀 ID
        league_id: 리그 ID
        fixture_id: 경기 ID
        upcoming: 예정된 경기 여부 (True=예정된 경기, False=지난 경기)

    Returns:
        검색 결과와 스포츠 데이터
    """
    query_params = {
        "date": date,
        "team_id": team_id,
        "league_id": league_id,
        "fixture_id": fixture_id,
        "upcoming": upcoming
    }
    # logging.info(f"Searching fixtures - params: {query_params}")

    try:
        fixtures = await get_fixtures(
            team_id=team_id,
            league_id=league_id,
            date=date,
            fixture_id=fixture_id,
            upcoming=upcoming
        )

        if not fixtures:
            if fixture_id:
                return {
                    "message": f"No fixture found with ID {fixture_id}.",
                    "fixtures": [],
                    "sports_data": {}
                }
            elif team_id:
                return {
                    "message": f"No fixtures found for team ID {team_id}." +
                               (f" on {date}" if date else "") +
                               (f" in league {league_id}" if league_id else ""),
                    "fixtures": [],
                    "sports_data": {}
                }
            else:
                return {
                    "message": "No fixtures found" +
                               (f" on {date}" if date else "") +
                               (f" in league {league_id}" if league_id else "") +
                               (f" {'upcoming' if upcoming else 'past'} fixtures" if not date else ""),
                    "fixtures": [],
                    "sports_data": {}
                }

        processed_data = {"fixtures": fixtures}

        return {
            "message": f"Found {len(fixtures)} fixtures matching your criteria.",
            "fixtures": [
                {
                    "id": fixture["fixture"]["id"],
                    "date": fixture["fixture"]["date"],
                    "status": fixture["fixture"]["status"]["long"],
                    "home_team": {
                        "id": fixture["teams"]["home"]["id"],
                        "name": fixture["teams"]["home"]["name"],
                        "logo": fixture["teams"]["home"].get("logo")
                    },
                    "away_team": {
                        "id": fixture["teams"]["away"]["id"],
                        "name": fixture["teams"]["away"]["name"],
                        "logo": fixture["teams"]["away"].get("logo")
                    },
                    "league": {
                        "id": fixture["league"]["id"],
                        "name": fixture["league"]["name"],
                        "country": fixture["league"].get("country")
                    },
                    "venue": fixture["fixture"].get("venue", {}).get("name", "Unknown Venue"),
                    "score": {
                        "home": fixture["goals"]["home"],
                        "away": fixture["goals"]["away"]
                    } if fixture["goals"]["home"] is not None else None
                }
                for fixture in fixtures[:10]  # Limit to top 10 fixtures
            ],
            "sports_data": preprocess_sports_data(processed_data)
        }

    except Exception as e:
        logging.error(f"Error searching fixtures: {str(e)}")
        return {
            "message": "Error searching fixtures. Please try again.",
            "fixtures": [],
            "sports_data": {}
        }

# 도구 생성
league_search_tool = StructuredTool.from_function(
    func=league_search,
    name="league_search",
    description="Search for sports leagues by name or country. Use when the user wants to find information about leagues.",
    coroutine=league_search
)

team_search_tool = StructuredTool.from_function(
    func=team_search,
    name="team_search",
    description="Search for sports teams by name and their upcoming fixtures. Use when the user wants to find information about teams.",
    coroutine=team_search
)

fixture_search_tool = StructuredTool.from_function(
    func=fixture_search,
    name="fixture_search",
    description="Search for sports fixtures (matches) by date, team, league, or fixture ID. Use when the user wants to find specific matches.",
    coroutine=fixture_search
)
