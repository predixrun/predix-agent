import json
import re
from datetime import datetime
from typing import Any

from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.config import logger, settings
from app.services.sports_service import get_fixtures, search_teams

MARKET_CREATION_PROMPT = """
You are creating a prediction market based on the user's request.
Extract the relevant details from the conversation and create a market with appropriate options.

For sports markets:
- Identify the teams or events involved
- Set clear prediction options (win/lose or draw)
- Consider the betting amount mentioned

Create a concise title and description for the market.
"""


async def extract_market_details(
        message: str
) -> dict[str, Any]:
    """
    Extract market creation details from a user message.

    Args:
        message: User message content

    Returns:
        Dictionary with market creation details
    """
    logger.info(f"Extracting market details from: {message[:50]}...")

    # Initialize LLM
    llm = ChatOpenAI(
        model="chatgpt-4o-latest",
        temperature=0.1,
        api_key=settings.OPENAI_API_KEY
    )

    # Market detail extraction system prompt
    market_prompt = [
        SystemMessage(content="""Extract market creation details from the user's message.

For sports markets, identify:
1. Teams involved
2. Event/match date
3. Betting amount
4. Type of prediction (who will win, etc.)

Format your response as JSON:
{
    "teams": ["Team A", "Team B"],
    "event_date": "YYYY-MM-DD" or "this Sunday" or null if not specified,
    "bet_amount": number (in SOL),
    "prediction_type": "match_winner" or other relevant type
}"""),
        {"role": "user", "content": message}
    ]

    try:
        market_result = await llm.ainvoke(market_prompt)
        market_content = market_result.content

        # Extract JSON from response
        json_str = re.search(r'({.*})', market_content.replace('\n', ' '), re.DOTALL)

        if json_str:
            try:
                market_info = json.loads(json_str.group(1))
                return market_info
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from market details extraction")

        # Default values if parsing fails
        return {
            "teams": [],
            "event_date": None,
            "bet_amount": 1.0,
            "prediction_type": "match_winner"
        }

    except Exception as e:
        logger.error(f"Error extracting market details: {e}")
        return {
            "teams": [],
            "event_date": None,
            "bet_amount": 1.0,
            "prediction_type": "match_winner"
        }


async def create_prediction_market(
        message: str,
        user_id: str,
) -> dict[str, Any]:
    """
    Create a prediction market based on a user message.

    Args:
        message: User message content
        user_id: ID of the user creating the market

    Returns:
        Dictionary with market data and assistant response
    """
    logger.info(f"Creating prediction market for user {user_id}: {message[:50]}...")

    # Extract market details
    market_info = await extract_market_details(message)

    # Get actual sports data
    teams_data = []
    fixtures_data = []

    for team_name in market_info.get("teams", []):
        if team_name:
            team_results = await search_teams(team_name)
            teams_data.extend(team_results)

    # Extract team IDs
    team_ids = [team["team"]["id"] for team in teams_data]

    # Get upcoming fixtures for these teams
    if team_ids:
        for team_id in team_ids:
            fixtures = await get_fixtures(team_id=team_id, upcoming=True)
            fixtures_data.extend(fixtures)

    # Find a fixture that matches the teams
    target_fixture = None

    if len(team_ids) >= 2 and fixtures_data:
        for fixture in fixtures_data:
            home_id = fixture["teams"]["home"]["id"]
            away_id = fixture["teams"]["away"]["id"]

            if home_id in team_ids and away_id in team_ids:
                target_fixture = fixture
                break

    # If we don't have a specific fixture but have teams, use the first fixture for one of the teams
    if not target_fixture and team_ids and fixtures_data:
        target_fixture = fixtures_data[0]

    # If we don't have any team data, use a default fixture
    if not target_fixture:
        fixtures = await get_fixtures(upcoming=True)
        if fixtures:
            target_fixture = fixtures[0]

    # Generate market details
    bet_amount_value = market_info.get("bet_amount", 1.0)
    bet_amount = float(bet_amount_value) if bet_amount_value is not None else 1.0

    if target_fixture:
        home_team = target_fixture["teams"]["home"]["name"]
        away_team = target_fixture["teams"]["away"]["name"]
        match_date = target_fixture["fixture"]["date"]
        fixture_id = target_fixture["fixture"]["id"]

        title = f"{home_team} vs {away_team} Match Prediction"
        description = f"Prediction market for the match between {home_team} and {away_team} on {match_date}"

        # Create selections based on home team
        selections = [
            {
                "name": f"{home_team} Win",
                "type": "win",
                "description": f"{home_team} will win the match"
            },
            {
                "name": f"{home_team} Draw/Lose",
                "type": "draw_lose",
                "description": f"{home_team} will draw or lose the match"
            }
        ]

        # Create market data structure
        market_data = {
            "creator_id": user_id,
            "title": title,
            "description": description,
            "type": "binary",
            "category": "sports",
            "amount": bet_amount,
            "currency": "SOL",
            "close_date": match_date,
            "created_at": datetime.now().isoformat()
        }

        # Create selections data
        selections_data = selections

        # Create event details
        event_data = {
            "type": "football_match",
            "fixture_id": fixture_id,
            "home_team": {
                "id": target_fixture["teams"]["home"]["id"],
                "name": home_team
            },
            "away_team": {
                "id": target_fixture["teams"]["away"]["id"],
                "name": away_team
            },
            "league": {
                "id": target_fixture["league"]["id"],
                "name": target_fixture["league"]["name"],
                "country": target_fixture["league"]["country"]
            },
            "start_time": match_date,
            "venue": {
                "name": target_fixture["fixture"].get("venue", {}).get("name", ""),
                "city": target_fixture["fixture"].get("venue", {}).get("city", "")
            }
        }
    else:
        # Generic fallback
        title = "Sports Prediction Market"
        description = "Prediction market for an upcoming sports event"

        team_names = market_info.get("teams", ["Team A", "Team B"])
        team_a = team_names[0] if len(team_names) > 0 else "Team A"

        selections = [
            {
                "name": f"{team_a} Win",
                "type": "win",
                "description": f"{team_a} will win"
            },
            {
                "name": f"{team_a} Draw/Lose",
                "type": "draw_lose",
                "description": f"{team_a} will draw or lose"
            }
        ]

        # Create market data structure
        market_data = {
            "creator_id": user_id,
            "title": title,
            "description": description,
            "type": "binary",
            "category": "sports",
            "amount": bet_amount,
            "currency": "SOL",
            "close_date": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }

        # Create selections data
        selections_data = selections

        # Create event details
        event_data = {
            "type": "football_match",
            "fixture_id": 0,
            "home_team": {
                "id": 0,
                "name": team_a
            },
            "away_team": {
                "id": 0,
                "name": team_names[1] if len(team_names) > 1 else "Team B"
            },
            "league": {
                "id": 0,
                "name": "Unknown League",
                "country": "Unknown"
            },
            "start_time": datetime.now().isoformat(),
            "venue": {
                "name": "Unknown Venue",
                "city": "Unknown City"
            }
        }

    # Prepare the full market info package for USER BE
    market_package = {
        "market": market_data,
        "selections": selections_data,
        "event": event_data
    }

    # Create response message
    from app.models.chat import Message
    response_message = Message(
        role="assistant",
        content=f"I've created a prediction market: '{title}'. Please select one of the options."
    )

    return {
        "messages": [response_message],
        "title": title,
        "description": description,
        "status": "draft",
        "selections": selections,
        "bet_amount": bet_amount,
        "context": {
            "user_id": user_id,
            "market_package": market_package
        }
    }


async def process_market_selection(
        selected_option: str,
        bet_amount: float = 1.0
) -> dict[str, Any]:
    """
    Process a user's selection of a market option.

    Args:
        selected_option: The option selected by the user
        bet_amount: The amount to bet

    Returns:
        Dictionary with response message
    """
    logger.info(f"Processing market selection: {selected_option} with amount {bet_amount}")

    from app.models.chat import Message
    if not selected_option:
        return {
            "messages": [
                Message(role="assistant", content="I couldn't understand your selection. Please try again.")
            ]
        }

    # Create betting amount request message
    response_message = Message(
        role="assistant",
        content=f"You've selected {selected_option} and the wager is {bet_amount} SOL. Proceed?"
    )

    return {
        "messages": [response_message]
    }


# Create tools
create_market_tool = StructuredTool.from_function(
    func=create_prediction_market,
    name="create_prediction_market",
    description="Create a prediction market (mentions creating a market, betting on a team, etc.)",
    coroutine=create_prediction_market
)

process_selection_tool = StructuredTool.from_function(
    func=process_market_selection,
    name="process_market_selection",
    description="If the user selects a market option or provides a betting amount",
    coroutine=process_market_selection
)
