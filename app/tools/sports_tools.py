import json
import re
from typing import Any

from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.config import logger, settings
from app.services.sports_service import get_fixtures, search_teams


async def extract_sports_query(
        message: str
) -> dict[str, Any]:
    """
    Extract sports query information from a message.

    Args:
        message: User message content

    Returns:
        Dictionary with teams, league, and time period
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
1. Team names
2. League/competition
3. Time period (upcoming matches, past results, etc.)

Format your response as JSON:
{
    "teams": ["Team Name 1", "Team Name 2"],
    "league": "League Name",
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
        return {"teams": [], "league": None, "time_period": "upcoming"}

    except Exception as e:
        logger.error(f"Error extracting sports query: {e}")
        return {"teams": [], "league": None, "time_period": "upcoming"}


async def get_sports_information(
        message: str
) -> dict[str, Any]:
    """
    Get sports information based on a user message.

    Args:
        message: User message content

    Returns:
        Dictionary with sports data and assistant response
    """
    logger.info(f"Getting sports information for: {message[:50]}...")

    # Extract query info
    query_info = await extract_sports_query(message)

    # Initialize sports data container
    sports_data = {"fixtures": [], "teams": []}

    # Get team information
    for team_name in query_info.get("teams", []):
        if team_name:
            team_results = await search_teams(team_name)
            sports_data["teams"].extend(team_results)

    # Get fixtures/matches
    team_ids = []
    for team in sports_data["teams"]:
        team_ids.append(team["team"]["id"])

    if team_ids:
        for team_id in team_ids:
            fixtures = await get_fixtures(
                team_id=team_id,
                upcoming=query_info.get("time_period", "upcoming") == "upcoming"
            )
            sports_data["fixtures"].extend(fixtures)
    else:
        # Get general fixtures if no specific team
        fixtures = await get_fixtures(
            upcoming=query_info.get("time_period", "upcoming") == "upcoming"
        )
        sports_data["fixtures"].extend(fixtures)

    # Generate response with sports data
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model="chatgpt-4o-latest",
            temperature=0.2,
            api_key=settings.OPENAI_API_KEY
        )

        response_prompt = [
            SystemMessage(content="""You are a helpful sports information assistant.
Provide a concise, informative response based on the sports data provided.

Format your response in a readable way with the key information highlighted.
If multiple fixtures/matches are available, focus on the most relevant ones.

Don't mention that you're using API data; just present the information as facts."""),
            {"role": "user", "content": f"User query: {message}\n\nSports data: {json.dumps(sports_data)}"}
        ]

        response = await llm.ainvoke(response_prompt)

        from app.models.chat import Message
        return {
            "messages": [
                Message(role="assistant", content=response.content)
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
