import uuid
from datetime import datetime
from operator import add
from typing import Annotated, Any

from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.config import logger, settings
from app.db.langgraph_store import checkpointer
from app.services.sports_service import get_fixtures, search_teams
from app.services.user_be_service import request_market_creation


# Message model for chat
class Message(TypedDict):
    role: str
    content: str


# Selection model for market options
class MarketSelection(TypedDict):
    id: str
    name: str
    type: str
    description: str | None


# Chat state model
class ChatState(TypedDict):
    # Conversation state
    messages: Annotated[list[Message], add]  # Using reducer to append messages

    # Market state
    market_id: str | None
    title: str | None
    description: str | None
    status: str | None
    selections: list[MarketSelection]
    selected_option: str | None
    selected_id: str | None
    bet_amount: float | None
    creator_id: str | None

    # Sports data
    sports_data: dict[str, Any] | None

    # Metadata
    current_node: str | None

    # Additional context that doesn't fit elsewhere
    context: dict[str, Any]


# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-latest",
    temperature=0.2,
    api_key=settings.OPENAI_API_KEY
)

# System prompts
GENERAL_SYSTEM_PROMPT = """
You are an AI assistant for the PrediX prediction market platform. 
Predix allows users to create and participate in prediction markets for sports events.

Your main tasks are:
1. Help users create prediction markets by detecting their intent
2. Answer questions about sports events and prediction markets
3. Provide general assistance

For market creation, detect when users want to create a prediction market for sports events.
For sports questions, provide information about teams, matches, and events.
"""

MARKET_CREATION_PROMPT = """
You are creating a prediction market based on the user's request.
Extract the relevant details from the conversation and create a market with appropriate options.

For sports markets:
- Identify the teams or events involved
- Set clear prediction options (win/lose or draw)
- Consider the betting amount mentioned

Create a concise title and description for the market.
"""


# Node functions for the graph
async def process_message(state: ChatState) -> dict[str, Any]:
    """
    Process a user message and determine next action.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"current_node": "end"}

    latest_message = messages[-1]

    # Skip if not a user message
    if latest_message["role"] != "user":
        return {"current_node": "end"}

    content = latest_message["content"].lower()

    # Convert messages to LangChain format
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))

    # Add system message if it's not already present
    if not any(isinstance(msg, SystemMessage) for msg in lc_messages):
        lc_messages.insert(0, SystemMessage(content=GENERAL_SYSTEM_PROMPT))

    # Detect intent with LLM
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Analyze the user's message and determine their intent.
        Respond with one of these exact categories:
        - MARKET_CREATION: If they want to create a prediction market
        - SPORTS_INFO: If they're asking about sports information
        - GENERAL_CHAT: For general questions or conversation

        Just respond with the category name, nothing else.
        """),
        ("human", content)
    ])

    intent_chain = intent_prompt | llm
    intent_result = await intent_chain.ainvoke({})
    intent = intent_result.content.strip().upper()

    logger.info(f"Detected intent: {intent}")

    if "MARKET_CREATION" in intent:
        return {"current_node": "create_market"}
    elif "SPORTS_INFO" in intent:
        return {"current_node": "sports_info"}
    else:
        # General chat response
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", GENERAL_SYSTEM_PROMPT),
            *[(msg.__class__.__name__.lower().replace("message", ""), msg.content) for msg in lc_messages if
              not isinstance(msg, SystemMessage)]
        ])

        chat_chain = chat_prompt | llm
        response = await chat_chain.ainvoke({})

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": response.content
                }
            ],
            "current_node": "end"
        }


async def get_sports_info(state: ChatState) -> dict[str, Any]:
    """
    Get sports information based on the user's query.
    """
    messages = state.get("messages", [])
    latest_message = messages[-1]["content"] if messages else ""

    # Extract query info with LLM
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Extract the sports information request from the user's message.
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

        If any field is not mentioned, leave it as null or empty list [].
        """),
        ("human", latest_message)
    ])

    query_chain = query_prompt | llm
    query_result = await query_chain.ainvoke({})

    # Clean the response to get valid JSON
    import json
    import re

    json_str = re.search(r'({.*})', query_result.content.replace('\n', ' '), re.DOTALL)
    if json_str:
        try:
            query_info = json.loads(json_str.group(1))
        except json.JSONDecodeError:
            query_info = {"teams": [], "league": None, "time_period": "upcoming"}
    else:
        query_info = {"teams": [], "league": None, "time_period": "upcoming"}

    # Get sports data based on extracted info
    sports_data = {"fixtures": [], "teams": []}

    # Search for team info
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
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful sports information assistant.
        Provide a concise, informative response based on the sports data provided.

        Format your response in a readable way with the key information highlighted.
        If multiple fixtures/matches are available, focus on the most relevant ones.

        Don't mention that you're using API data; just present the information as facts.
        """),
        ("human", f"User query: {latest_message}\n\nSports data: {json.dumps(sports_data)}"),
    ])

    response_chain = response_prompt | llm
    response = await response_chain.ainvoke({})

    return {
        "messages": [
            {
                "role": "assistant",
                "content": response.content
            }
        ],
        "sports_data": sports_data,
        "current_node": "end"
    }


async def create_market(state: ChatState) -> dict[str, Any]:
    """
    Create a new prediction market based on the user's request.
    """
    messages = state.get("messages", [])
    latest_message = messages[-1]["content"] if messages else ""
    user_id = state.get("creator_id")

    # Extract market details with LLM
    market_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Extract market creation details from the user's message.

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
        }
        """),
        ("human", latest_message)
    ])

    market_chain = market_prompt | llm
    market_result = await market_chain.ainvoke({})

    # Clean the response to get valid JSON
    import json
    import re

    json_str = re.search(r'({.*})', market_result.content.replace('\n', ' '), re.DOTALL)
    if json_str:
        try:
            market_info = json.loads(json_str.group(1))
        except json.JSONDecodeError:
            market_info = {
                "teams": [],
                "event_date": None,
                "bet_amount": 1.0,
                "prediction_type": "match_winner"
            }
    else:
        market_info = {
            "teams": [],
            "event_date": None,
            "bet_amount": 1.0,
            "prediction_type": "match_winner"
        }

    # Get actual sports data to create accurate market
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
    market_id = str(uuid.uuid4())
    bet_amount = float(market_info.get("bet_amount", 1.0))

    if target_fixture:
        home_team = target_fixture["teams"]["home"]["name"]
        away_team = target_fixture["teams"]["away"]["name"]
        match_date = target_fixture["fixture"]["date"]

        title = f"{home_team} vs {away_team} Match Prediction"
        description = f"Prediction market for the match between {home_team} and {away_team} on {match_date}"

        # Create selections based on home team
        selection_win_id = str(uuid.uuid4())
        selection_draw_lose_id = str(uuid.uuid4())

        selections = [
            {
                "id": selection_win_id,
                "name": f"{home_team} Win",
                "type": "win",
                "description": f"{home_team} will win the match"
            },
            {
                "id": selection_draw_lose_id,
                "name": f"{home_team} Draw/Lose",
                "type": "draw_lose",
                "description": f"{home_team} will draw or lose the match"
            }
        ]

        # Create market data to request from USER BE
        event_details = {
            "fixture_id": target_fixture["fixture"]["id"],
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            "league": target_fixture["league"]["name"],
            "country": target_fixture["league"]["country"]
        }
    else:
        # Generic fallback
        title = "Sports Prediction Market"
        description = "Prediction market for an upcoming sports event"

        selection_win_id = str(uuid.uuid4())
        selection_draw_lose_id = str(uuid.uuid4())

        team_names = market_info.get("teams", ["Team A", "Team B"])
        team_a = team_names[0] if len(team_names) > 0 else "Team A"

        selections = [
            {
                "id": selection_win_id,
                "name": f"{team_a} Win",
                "type": "win",
                "description": f"{team_a} will win"
            },
            {
                "id": selection_draw_lose_id,
                "name": f"{team_a} Draw/Lose",
                "type": "draw_lose",
                "description": f"{team_a} will draw or lose"
            }
        ]

        event_details = {}

    # Create response message
    response_message = {
        "role": "assistant",
        "content": f"I've created a prediction market: '{title}'. Please select one of the options."
    }

    return {
        "messages": [response_message],
        "market_id": market_id,
        "title": title,
        "description": description,
        "status": "draft",
        "selections": selections,
        "bet_amount": bet_amount,
        "current_node": "wait_for_selection",
        "context": {
            "user_id": user_id,
            "event_details": event_details
        }
    }


async def process_selection(state: ChatState) -> dict[str, Any]:
    """
    Process a user's selection of a market option.
    """
    selected_id = state.get("selected_id")

    if not selected_id:
        return {
            "messages": [{
                "role": "assistant",
                "content": "I couldn't understand your selection. Please try again."
            }],
            "current_node": "wait_for_selection"
        }

    selections = state.get("selections", [])
    bet_amount = state.get("bet_amount", 1.0)

    # Find the selected option
    selected_option = next((s["name"] for s in selections if s["id"] == selected_id), None)

    if not selected_option:
        return {
            "messages": [{
                "role": "assistant",
                "content": "Invalid selection. Please select a valid option."
            }],
            "current_node": "wait_for_selection"
        }

    # Create confirmation message
    response_message = {
        "role": "assistant",
        "content": f"You've selected {selected_option} and the wager is {bet_amount} SOL. Proceed?"
    }

    return {
        "messages": [response_message],
        "selected_option": selected_option,
        "current_node": "wait_for_confirmation"
    }


async def process_confirmation(state: ChatState) -> dict[str, Any]:
    """
    Process a user's confirmation to create a market.
    """
    context = state.get("context", {})
    confirmed = context.get("confirmed", False)

    market_id = state.get("market_id")
    title = state.get("title")
    creator_id = state.get("creator_id")
    selections = state.get("selections", [])
    selected_id = state.get("selected_id")
    bet_amount = state.get("bet_amount", 1.0)
    description = state.get("description", "")
    event_details = context.get("event_details", {})

    if not confirmed:
        # User cancelled
        return {
            "messages": [{
                "role": "assistant",
                "content": "Market creation cancelled."
            }],
            "status": "cancelled",
            "current_node": "end"
        }

    # User confirmed, request market creation from USER BE
    market_data = {
        "id": market_id,
        "creator_id": creator_id,
        "title": title,
        "description": description,
        "selections": selections,
        "bet_amount": bet_amount,
        "selected_id": selected_id,
        "event_details": event_details,
        "created_at": datetime.now().isoformat()
    }

    # Request market creation from USER BE
    market_response = await request_market_creation(market_data)

    response_message = {
        "role": "assistant",
        "content": "Market is open and this participation link to share"
    }

    return {
        "messages": [response_message],
        "status": "open",
        "current_node": "end"
    }


# Router function to determine the next node
def router(state: ChatState) -> str:
    current_node = state.get("current_node")

    if current_node == "create_market":
        return "create_market"
    elif current_node == "sports_info":
        return "sports_info"
    elif current_node == "wait_for_selection":
        return "wait_for_selection"
    elif current_node == "wait_for_confirmation":
        return "wait_for_confirmation"
    else:
        return "end"


# Create the state graph
def create_chat_graph() -> StateGraph:
    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("process_message", process_message)
    workflow.add_node("create_market", create_market)
    workflow.add_node("sports_info", get_sports_info)
    workflow.add_node("wait_for_selection", process_selection)
    workflow.add_node("wait_for_confirmation", process_confirmation)

    # Add edges
    workflow.add_edge(START, "process_message")
    workflow.add_conditional_edges(
        "process_message",
        router,
        {
            "create_market": "create_market",
            "sports_info": "sports_info",
            "wait_for_selection": "wait_for_selection",
            "wait_for_confirmation": "wait_for_confirmation",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "create_market",
        router,
        {
            "wait_for_selection": "wait_for_selection",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "sports_info",
        router,
        {
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "wait_for_selection",
        router,
        {
            "wait_for_selection": "wait_for_selection",
            "wait_for_confirmation": "wait_for_confirmation",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "wait_for_confirmation",
        router,
        {
            "wait_for_confirmation": "wait_for_confirmation",
            "end": END
        }
    )

    return workflow


# Singleton pattern for the compiled graph
_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        workflow = create_chat_graph()
        _compiled_graph = workflow.compile(checkpointer=checkpointer)
    return _compiled_graph


# Initialize the graph
def init_graph():
    get_compiled_graph()
