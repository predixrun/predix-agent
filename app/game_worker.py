import logging
from typing import Tuple, Any, Dict, Optional, Callable, Awaitable, Union

from game_sdk.game.worker import Worker
from game_sdk.game.custom_types import Function, Argument, FunctionResult, FunctionResultStatus

from app.config import settings
from app.services import sports_service

game_api_key = settings.GAME_API_KEY
if not game_api_key:
    raise ValueError("GAME_API_KEY environment variable not set.")

LLAMA_MODEL_NAME = "Llama-3.1-405B-Instruct"

# --- State Management Function ---
def get_football_worker_state(function_result: Optional[FunctionResult], current_state: Optional[dict]) -> dict:
    """
    Manages the state for the football worker.
    Stores the result of the last executed function.
    """
    if current_state is None:
        # Initial state
        new_state = {"last_search_result": None}
    else:
        new_state = current_state.copy() # Important: Copy the state

    if function_result and function_result.info:
        new_state["last_search_result"] = function_result.info
        logging.debug(f"FootballWorker state updated with result: {function_result.info}")
    elif function_result and function_result.action_status == FunctionResultStatus.FAILED:
         new_state["last_search_result"] = {"error": str(function_result)} # Store error message-> 오류 발생할것.
         logging.warning(f"FootballWorker state updated with error: {str(function_result)}")
    # else: No function result or no info, keep state as is

    return new_state

# --- Executable Wrappers for GAME Functions ---

async def game_team_search_executable(name: str, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
    """GAME Function Executable: Searches for teams."""
    logging.info(f"GAME Worker executing team search: name='{name}'")
    try:
        result_data = await sports_service.search_teams(name)
        # The service function returns a dict with 'message', 'teams', 'fixtures', 'sports_data'
        return FunctionResultStatus.DONE, "Teams/Fixtures found.", {"search_result": result_data}

    except Exception as e:
        logging.error(f"Error in game_team_search_executable: {e}", exc_info=True)
        return FunctionResultStatus.FAILED, f"Error searching teams: {e}", {"error": str(e)}

async def game_fixture_search_executable(
    date: Optional[str] = None,
    team_id: Optional[int] = None,
    league_id: Optional[int] = None,
    fixture_id: Optional[int] = None,
    upcoming: bool = True,
    **kwargs
) -> Tuple[FunctionResultStatus, str, dict]:
    """GAME Function Executable: Searches for fixtures."""
    logging.info(f"GAME Worker executing fixture search: date='{date}', team_id={team_id}, league_id={league_id}, fixture_id={fixture_id}, upcoming={upcoming}")
    try:
        result_data = await sports_service.get_fixtures(
            date=date, team_id=team_id, league_id=league_id, fixture_id=fixture_id, upcoming=upcoming
        )
        # The service function returns a dict with 'message', 'fixtures', 'sports_data'
        if result_data:
            return FunctionResultStatus.DONE, "Fixtures found.", {"search_result": result_data}
        else:
            return FunctionResultStatus.DONE, "No fixtures found.", {"search_result": result_data}
    except Exception as e:
        logging.error(f"Error in game_fixture_search_executable: {e}", exc_info=True)
        return FunctionResultStatus.FAILED, f"Error searching fixtures: {e}", {"error": str(e)}


def sync_wrapper(async_fn: Callable[..., Awaitable[Tuple[FunctionResultStatus, str, dict]]]) -> Callable[..., Tuple[FunctionResultStatus, str, dict]]:
    """Create a synchronous wrapper for an async function."""
    import asyncio
    
    def wrapper(*args, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(async_fn(*args, **kwargs))
    
    return wrapper

sync_game_team_search = sync_wrapper(game_team_search_executable)
sync_game_fixture_search = sync_wrapper(game_fixture_search_executable)

game_team_search_fn = Function(
    fn_name="search_teams",
    fn_description="Search for football teams by name. Returns team info and upcoming fixtures.",
    args=[
        Argument(name="name", type="string", description="Team name to search for in English"),
    ],
    executable=sync_game_team_search
)

game_fixture_search_fn = Function(
    fn_name="search_fixtures",
    fn_description="Search for football fixtures (matches) by date, team ID, or specific fixture ID. Specify 'upcoming' for future (default) or past matches.",
    args=[
        Argument(name="date", type="string", description="Specific date in YYYY-MM-DD format (optional, you can set it to empty str)"),
        Argument(name="fixture_id", type="integer", description="Specific fixture ID to retrieve (optional, If you don't know, set it to '' )"),
        Argument(name="upcoming", type="boolean", description="True for upcoming (default), False for past fixtures"),
    ],
    executable=sync_game_fixture_search
)

# Action Space
football_action_space = [
    game_team_search_fn,
    game_fixture_search_fn,
]

# Mock worker implementation for when the real worker cannot be initialized
class MockWorker:
    def __init__(self):
        self.state = {"error": "Worker initialization failed"}
        logging.warning("Using MockWorker because the real GAME Worker could not be initialized")
    
    def run(self, input_text: str) -> Any:
        logging.warning(f"MockWorker received request: {input_text}")
        return {"message": "The football information service is currently unavailable. Please try again later."}

# Try to instantiate the GAME Worker, but fall back to mock if it fails
try:
    # Instantiate the GAME Worker
    football_worker = Worker(
        api_key=game_api_key,
        description="An expert agent specialized in retrieving information about football teams, and fixtures using real-time data.",
        instruction="Use the available functions to find the specific football information requested.",
        get_state_fn=get_football_worker_state,
        action_space=football_action_space,
        model_name=LLAMA_MODEL_NAME
    )
    logging.info(f"GAME Football Worker initialized with model {LLAMA_MODEL_NAME} and {len(football_action_space)} actions.")
except Exception as e:
    logging.error(f"Failed to initialize GAME Worker due to 429: {e}", exc_info=True)
    football_worker = MockWorker()


class AsyncWorkerWrapper:
    """Wrapper class to make worker.run() work properly with async executables."""
    
    def __init__(self, worker: Worker):
        self.worker = worker
        
    async def run_async(self, input_text: str) -> Any:
        """Run the worker with the given input text."""
        try:
            return self.worker.run(input_text)
        except Exception as e:
            logging.error(f"Error running worker: {e}", exc_info=True)
            return {"error": str(e), "message": "Failed to process your request. The football information service is currently unavailable."}
    
    @property
    def state(self):
        """Access the worker's state if available"""
        if hasattr(self.worker, 'state'):
            return self.worker.state
        return {"error": "State not available"}


# Create the async wrapper with error handling
try:
    async_football_worker = AsyncWorkerWrapper(football_worker)
except Exception as e:
    logging.error(f"Failed to create AsyncWorkerWrapper: {e}", exc_info=True)
    # Create a minimal async worker that just returns error messages
    class MinimalAsyncWorker:
        async def run_async(self, input_text: str) -> Any:
            return {"error": "Service unavailable", "message": "The football information service is currently unavailable."}
        
        @property
        def state(self):
            return {"error": "Worker initialization failed", "last_search_result": None}
        
        @property
        def worker(self):
            return self  # Self-reference for compatibility
    
    async_football_worker = MinimalAsyncWorker()


if __name__ == "__main__":
    result = football_worker.run("Find any upcoming fixtures")
    print(f"Worker State after run: {football_worker.state}")