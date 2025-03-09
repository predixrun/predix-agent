import json
import re
from datetime import datetime
from typing import Any

from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.config import logger, settings
from app.services.sports_service import get_fixtures, get_leagues, preprocess_sports_data, search_teams


async def extract_sports_query(
        message: str
) -> dict[str, Any]:
    """
    Extract sports query information from a message.

    Args:
        message: User message content

    Returns:
        Dictionary with query parameters
    """
    logger.info(f"Extracting sports query from: {message[:50]}...")

    # Initialize LLM
    llm = ChatOpenAI(
        model="chatgpt-4o-latest",
        temperature=0.1,
        api_key=settings.OPENAI_API_KEY
    )

    # Query extraction system prompt
    query_prompt = [
        SystemMessage(content="""Extract the sports information request from the user's message.
Identify:
1. Team names (if any)
2. League/competition (if any)
3. Specific dates or time periods (e.g., "this Sunday", "March 10", etc.)
4. Fixture ID (if mentioned)
5. Query type: 
   - "team_search" (looking for team info)
   - "fixture_search" (looking for match/fixture info)
   - "date_search" (looking for matches on specific dates)
   - "specific_fixture" (looking for details about a specific match with ID)

Format your response as JSON:
{
    "teams": ["Team Name 1", "Team Name 2"],
    "league": "League Name",
    "date": "YYYY-MM-DD" or "specific date description",
    "fixture_id": 123456,
    "query_type": "team_search"/"fixture_search"/"date_search"/"specific_fixture",
    "time_period": "upcoming" or "past"
}

If any field is not mentioned, leave it as null or empty list []."""),
        {"role": "user", "content": message}
    ]

    try:
        query_result = await llm.ainvoke(query_prompt)
        query_content = query_result.content

        # Extract JSON from response
        json_str = re.search(r'({.*})', query_content.replace('\n', ' '), re.DOTALL)

        if json_str:
            try:
                query_info = json.loads(json_str.group(1))
                return query_info
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from sports query extraction")

        # Default values if parsing fails
        return {
            "teams": [],
            "league": None,
            "date": None,
            "fixture_id": None,
            "query_type": "fixture_search",
            "time_period": "upcoming"
        }

    except Exception as e:
        logger.error(f"Error extracting sports query: {e}")
        return {
            "teams": [],
            "league": None,
            "date": None,
            "fixture_id": None,
            "query_type": "fixture_search",
            "time_period": "upcoming"
        }


async def parse_date(date_str: str) -> str | None:
    """
    Parse a date string into a YYYY-MM-DD format.

    Args:
        date_str: Date string (e.g., "this Sunday", "March 10")

    Returns:
        Formatted date string or None
    """
    if not date_str:
        return None

    try:
        # Use LLM to parse the date
        llm = ChatOpenAI(
            model="chatgpt-4o-latest",
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )

        date_prompt = [
            SystemMessage(content="""Convert the following date description to a YYYY-MM-DD format.
If the date is relative (like "this Sunday", "tomorrow", etc.), calculate it based on today's date.
Today's date is {} (YYYY-MM-DD format).
Just return the date in YYYY-MM-DD format, nothing else.""".format(
                datetime.now().strftime("%Y-%m-%d")
            )),
            {"role": "user", "content": date_str}
        ]

        date_result = await llm.ainvoke(date_prompt)
        parsed_date = date_result.content.strip()

        # Validate date format
        datetime.strptime(parsed_date, "%Y-%m-%d")
        return parsed_date
    except Exception as e:
        logger.error(f"Error parsing date '{date_str}': {e}")
        return None


async def get_sports_information(
        message: str
) -> dict[str, Any]:
    """
    Get sports information based on a user message using the real sports API.

    Args:
        message: User message content

    Returns:
        Dictionary with sports data and assistant response
    """
    logger.info(f"Getting sports information for: {message[:50]}...")

    # Extract query info
    query_info = await extract_sports_query(message)
    logger.info(f"Extracted query info: {query_info}")

    # Initialize sports data container
    sports_data = {"fixtures": [], "teams": []}
    error_msg = None

    try:
        # Handle different query types
        query_type = query_info.get("query_type", "fixture_search")

        # Parse date if provided
        date_str = query_info.get("date")
        parsed_date = await parse_date(date_str) if date_str else None

        # Process based on query type
        if query_type == "specific_fixture" and query_info.get("fixture_id"):
            # Get specific fixture by ID
            fixture_id = query_info.get("fixture_id")
            fixtures = await get_fixtures(fixture_id=fixture_id)
            sports_data["fixtures"] = fixtures

        elif query_type == "team_search" and query_info.get("teams"):
            # Get team information
            for team_name in query_info.get("teams", []):
                if team_name:
                    team_results = await search_teams(team_name)
                    sports_data["teams"].extend(team_results)

            # Get fixtures for teams if found
            team_ids = [team["team"]["id"] for team in sports_data["teams"]]
            if team_ids:
                for team_id in team_ids:
                    fixtures = await get_fixtures(
                        team_id=team_id,
                        date=parsed_date,
                        upcoming=query_info.get("time_period", "upcoming") == "upcoming"
                    )
                    sports_data["fixtures"].extend(fixtures)

        elif query_type == "date_search" and parsed_date:
            # Get fixtures by date
            fixtures = await get_fixtures(date=parsed_date)
            sports_data["fixtures"] = fixtures

        else:
            # Default to general fixture search
            # Get fixtures based on available parameters
            team_names = query_info.get("teams", [])
            league_name = query_info.get("league")

            # If teams are specified, get their IDs first
            team_ids = []
            if team_names:
                for team_name in team_names:
                    if team_name:
                        team_results = await search_teams(team_name)
                        sports_data["teams"].extend(team_results)
                        team_ids.extend([team["team"]["id"] for team in team_results])

            # Get league ID if specified
            league_id = None
            if league_name:
                leagues = await get_leagues(search=league_name)
                if leagues:
                    # Use the first matching league
                    league_id = leagues[0]["league"]["id"]

            # Get fixtures based on parameters
            if team_ids:
                for team_id in team_ids:
                    fixtures = await get_fixtures(
                        team_id=team_id,
                        league_id=league_id,
                        date=parsed_date,
                        upcoming=query_info.get("time_period", "upcoming") == "upcoming"
                    )
                    sports_data["fixtures"].extend(fixtures)
            elif league_id:
                fixtures = await get_fixtures(
                    league_id=league_id,
                    date=parsed_date,
                    upcoming=query_info.get("time_period", "upcoming") == "upcoming"
                )
                sports_data["fixtures"] = fixtures
            elif parsed_date:
                fixtures = await get_fixtures(
                    date=parsed_date,
                    upcoming=query_info.get("time_period", "upcoming") == "upcoming"
                )
                sports_data["fixtures"] = fixtures
            else:
                # Generic fixtures search
                fixtures = await get_fixtures(
                    upcoming=query_info.get("time_period", "upcoming") == "upcoming"
                )
                sports_data["fixtures"] = fixtures

        # Check if we have valid data
        if not sports_data["fixtures"] and not sports_data["teams"]:
            error_msg = "I couldn't find any sports information matching your query. Could you please provide more details or try a different search?"
        else:
            # Preprocess data to make it more manageable for LLM
            sports_data = preprocess_sports_data(sports_data)

    except Exception as e:
        logger.error(f"Error getting sports data: {e}")
        error_msg = "I encountered an error while retrieving sports information. Please try again with a different query."

    # Generate response with sports data
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model="chatgpt-4o-latest",
            temperature=0.2,
            api_key=settings.OPENAI_API_KEY
        )

        if error_msg:
            response_content = error_msg
        else:
            # todo: fixture id 처리 고도화
            response_prompt = [
                SystemMessage(content="""You are a helpful sports information assistant.
Provide a concise, informative response based on the sports data provided.

Format your response in a readable way with the key information highlighted.
If multiple fixtures/matches are available, focus on the most relevant ones (up to 5).
For fixtures, include:
- Match: Team A vs Team B
- Date and time
- Competition/League
- Venue (if available)
- Status (upcoming or result with score if finished)
- fixture id

Don't mention that you're using API data; just present the information as facts."""),
                {"role": "user", "content": f"User query: {message}\n\nSports data: {json.dumps(sports_data)}"}
            ]

            response = await llm.ainvoke(response_prompt)
            response_content = response.content

        from app.models.chat import Message
        return {
            "messages": [
                Message(role="assistant", content=response_content)
            ],
            "sports_data": sports_data
        }

    except Exception as e:
        logger.error(f"Error generating sports response: {e}")
        from app.models.chat import Message
        return {
            "messages": [
                Message(role="assistant", content="I'm having trouble retrieving sports information. Please try again.")
            ],
            "sports_data": sports_data
        }


# Create tools
sports_info_tool = StructuredTool.from_function(
    func=get_sports_information,
    name="get_sports_information",
    description="Get sports information based on a user message",
    coroutine=get_sports_information
)
