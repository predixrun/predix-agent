import asyncio
import logging
from typing import Optional, Dict, Any

from langchain_core.tools import StructuredTool

from app.game_worker import async_football_worker
from app.tools.sports_tools import league_search, team_search, fixture_search

def call_football_worker_sync(
    action: str,
    search_term: Optional[str] = None,
    country: Optional[str] = None,
    team_name: Optional[str] = None,
    league_id: Optional[int] = None,
    date: Optional[str] = None,
    team_id: Optional[int] = None,
    fixture_id: Optional[int] = None,
    upcoming: bool = True
) -> Dict[str, Any]:
    """
    Call the GAME football worker.
    """
    instruction = f"Perform action '{action}'. "
    details = []
    if search_term: details.append(f"search_term='{search_term}'")
    if country: details.append(f"country='{country}'")
    if team_name: details.append(f"team_name='{team_name}'")
    if league_id is not None: details.append(f"league_id={league_id}")
    if date: details.append(f"date='{date}'")
    if team_id is not None: details.append(f"team_id={team_id}")
    if fixture_id is not None: details.append(f"fixture_id={fixture_id}")
    if action == "search_fixtures": details.append(f"upcoming={upcoming}") # Only relevant for fixture search

    if details:
        instruction += "Parameters: " + ", ".join(details)
    else:
         instruction += "No specific parameters provided." # Should ideally not happen if action is valid

    logging.info(f"LangGraph Tool calling GAME Worker with instruction: {instruction}")

    try:
        # GAME Worker
        async_football_worker.run_async(instruction)
        worker_state = async_football_worker.state
        result = worker_state.get("last_search_result", {})

        logging.info(f"GAME Worker execution finished. Result from state: {result}")

        if isinstance(result, dict) and "error" in result:
             logging.error(f"GAME Worker returned an error: {result['error']}")
             return c_sports_tools(action, search_term, country, team_name, league_id, date, team_id, fixture_id, upcoming)

        search_data = result.get("search_result", {})
        return search_data

    except Exception as e:
        logging.error(f"Error calling GAME football worker: {e}", exc_info=True)
        return c_sports_tools(action, search_term, country, team_name, league_id, date, team_id, fixture_id, upcoming)

def run_async_safely(coroutine):
    """Helper function to run async coroutines in a way that works with or without an existing event loop"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run(coroutine)
        else:
            return loop.run_until_complete(coroutine)
    except RuntimeError:
        return asyncio.run(coroutine)

def c_sports_tools(
    action: str,
    search_term: Optional[str] = None,
    country: Optional[str] = None,
    team_name: Optional[str] = None,
    league_id: Optional[int] = None,
    date: Optional[str] = None,
    team_id: Optional[int] = None,
    fixture_id: Optional[int] = None,
    upcoming: bool = True
) -> Dict[str, Any]:
    try:
        if action == "search_leagues":
            coro = league_search(search=search_term, country=country)
        elif action == "search_teams":
            coro = team_search(name=team_name or search_term, league_id=league_id)
        elif action == "search_fixtures":
            coro = fixture_search(
                date=date,
                team_id=team_id,
                league_id=league_id,
                fixture_id=fixture_id,
                upcoming=upcoming
            )
        else:
            return {
                "message": f"Unknown action: {action}",
                "sports_data": {},
                "error": f"Unknown action: {action}"
            }
        
        # Run the async function safely
        return run_async_safely(coro)
    except Exception as e:
        logging.error(f"Error in fallback to sports_tools: {e}", exc_info=True)
        return {
            "message": f"Both worker and fallback failed: {e}",
            "sports_data": {},
            "error": str(e)
        }

football_information_retriever_tool = StructuredTool.from_function(
    func=call_football_worker_sync,
    name="football_information_retriever",
    description=(
        "Use this tool to retrieve information about football (soccer) leagues, teams, or fixtures. "
        "You MUST specify the 'action' to perform: 'search_leagues', 'search_teams', or 'search_fixtures'. "
        "Provide relevant parameters based on the action: "
        "'search_leagues' needs 'search_term' (league name) and/or 'country'. "
        "'search_teams' needs 'team_name' and optionally 'league_id'. "
        "'search_fixtures' needs criteria like 'date', 'team_id', 'league_id', 'fixture_id', or 'upcoming' (True/False)."
    ),
)