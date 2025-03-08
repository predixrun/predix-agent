import uuid
from operator import add
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.db.langgraph_store import checkpointer


# Message model for chat
class Message(TypedDict):
    role: str
    content: str


# Selection model for market options
class Selection(TypedDict):
    id: str
    name: str
    type: str
    description: str | None


# Market state model
class MarketState(TypedDict):
    # Conversation state
    messages: Annotated[list[Message], add]  # Using reducer to append messages

    # Market state
    market_id: str | None
    title: str | None
    description: str | None
    status: str | None
    selections: list[Selection]
    selected_option: str | None
    selected_id: str | None
    bet_amount: float | None
    creator_id: str | None

    # Metadata
    current_node: str | None

    # Additional context that doesn't fit elsewhere
    context: dict[str, Any]


# Node functions for the graph
def process_message(state: MarketState) -> dict[str, Any]:
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

    # Detect if this is a market creation request
    if any(keyword in content for keyword in ["prediction", "market", "bet", "wager"]):
        return {"current_node": "create_market"}

    # Default response if not a special command
    response = {
        "messages": [
            {
                "role": "assistant",
                "content": "I can help you create prediction markets. Try asking something like 'Create a prediction for Manchester City vs United game this Sunday with 12 SOL wager'."
            }
        ],
        "current_node": "end"
    }

    return response


def create_market(state: MarketState) -> dict[str, Any]:
    """
    Create a new prediction market based on the user's request.
    """
    messages = state.get("messages", [])
    latest_message = messages[-1]["content"].lower() if messages else ""

    # Extract market details (simplified implementation)
    market_id = str(uuid.uuid4())
    title = ""
    description = ""
    selections = []
    bet_amount = 1.0  # Default
    status = "draft"

    # Simple parsing for demo
    if "manchester city" in latest_message and "united" in latest_message:
        title = "Which team will be the winner of this Sunday's derby?"
        description = "Prediction market for Manchester City vs Manchester United match"

        selections = [
            {
                "id": str(uuid.uuid4()),
                "name": "Manchester City Win",
                "type": "win",
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Manchester City Draw/Lose",
                "type": "draw_lose",
            }
        ]

        # Extract bet amount if specified
        if "wager" in latest_message or "bet" in latest_message:
            import re
            amount_match = re.search(r'(\d+(?:\.\d+)?)\s*sol', latest_message)
            if amount_match:
                bet_amount = float(amount_match.group(1))

    # For crypto markets or other types, add similar logic here
    elif "bitcoin" in latest_message or "btc" in latest_message:
        title = "Will Bitcoin price increase by the end of the week?"
        description = "Prediction market for Bitcoin price movement"

        selections = [
            {
                "id": str(uuid.uuid4()),
                "name": "Price Increase",
                "type": "win",
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Price Decrease or No Change",
                "type": "draw_lose",
            }
        ]

        # Extract bet amount
        import re
        amount_match = re.search(r'(\d+(?:\.\d+)?)\s*sol', latest_message)
        if amount_match:
            bet_amount = float(amount_match.group(1))

    # Generic fallback
    else:
        title = "Prediction Market"
        description = "Market created from user request"

        selections = [
            {
                "id": str(uuid.uuid4()),
                "name": "Option A",
                "type": "win",
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Option B",
                "type": "draw_lose",
            }
        ]

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
        "status": status,
        "selections": selections,
        "bet_amount": bet_amount,
        "current_node": "wait_for_selection"
    }


def process_selection(state: MarketState) -> dict[str, Any]:
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


def process_confirmation(state: MarketState) -> dict[str, Any]:
    """
    Process a user's confirmation to create a market.
    """
    context = state.get("context", {})
    confirmed = context.get("confirmed", False)

    market_id = state.get("market_id")
    title = state.get("title")

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

    # User confirmed, finalize market
    share_url = f"http://predix/market/{market_id}"

    response_message = {
        "role": "assistant",
        "content": f"Market is open and this participation link to share: {share_url}"
    }

    return {
        "messages": [response_message],
        "status": "open",
        "current_node": "end"
    }


# Router function to determine the next node
def router(state: MarketState) -> str:
    current_node = state.get("current_node")

    if current_node == "create_market":
        return "create_market"
    elif current_node == "wait_for_selection":
        return "wait_for_selection"
    elif current_node == "wait_for_confirmation":
        return "wait_for_confirmation"
    else:
        return "end"


# Create the state graph
def create_market_graph() -> StateGraph:
    workflow = StateGraph(MarketState)

    # Add nodes
    workflow.add_node("process_message", process_message)
    workflow.add_node("create_market", create_market)
    workflow.add_node("wait_for_selection", process_selection)
    workflow.add_node("wait_for_confirmation", process_confirmation)

    # Add edges
    workflow.add_edge(START, "process_message")
    workflow.add_conditional_edges(
        "process_message",
        router,
        {
            "create_market": "create_market",
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
        workflow = create_market_graph()
        _compiled_graph = workflow.compile(checkpointer=checkpointer)
    return _compiled_graph


# Initialize the graph
def init_graph():
    get_compiled_graph()
