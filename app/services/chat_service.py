import uuid

from app.config import logger
from app.db import langgraph_store
from app.graph.chat_graph import get_compiled_graph
from app.models.chat import ChatResponse


async def process_chat_message(
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        wallet_address: str | None = None
) -> ChatResponse:
    """
    Process a chat message from a user using LangGraph.

    Args:
        user_id: ID of the user
        message: The message content
        conversation_id: Optional ID of the conversation
        wallet_address: Optional wallet address of the user

    Returns:
        ChatResponse object with appropriate response
    """
    # Generate conversation ID if not provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    # Log the incoming message
    logger.info(f"Received message from user {user_id} in conversation {conversation_id}")

    # Get the compiled graph
    graph = get_compiled_graph()

    # Create initial state or get existing state
    try:
        # Try to get existing state
        config = {"configurable": {"thread_id": conversation_id}}
        current_state = langgraph_store.get_conversation_state(conversation_id)

        # If we have no state or empty state, initialize it
        if not current_state:
            current_state = {
                "messages": [],
                "market_id": None,
                "title": None,
                "description": None,
                "status": None,
                "selections": [],
                "selected_option": None,
                "selected_id": None,
                "bet_amount": None,
                "creator_id": user_id,
                "sports_data": None,
                "current_node": None,
                "context": {"user_id": user_id, "wallet_address": wallet_address}
            }
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        # Initialize new state if error
        current_state = {
            "messages": [],
            "market_id": None,
            "title": None,
            "description": None,
            "status": None,
            "selections": [],
            "selected_option": None,
            "selected_id": None,
            "bet_amount": None,
            "creator_id": user_id,
            "sports_data": None,
            "current_node": None,
            "context": {"user_id": user_id, "wallet_address": wallet_address}
        }

    # Add user message to conversation
    user_message = {"role": "user", "content": message}

    # Update messages with new user message
    if "messages" in current_state:
        messages = current_state["messages"] + [user_message]
    else:
        messages = [user_message]

    # Update the state with the new message
    input_state = {**current_state, "messages": messages}

    # Run the graph
    config = {"configurable": {"thread_id": conversation_id}}

    try:
        result = graph.invoke(input_state, config)

        # Get the last assistant message
        assistant_messages = [msg for msg in result.get("messages", [])
                              if msg["role"] == "assistant"]

        if assistant_messages:
            latest_message = assistant_messages[-1]["content"]
        else:
            latest_message = "I didn't get a response. Please try again."

        # Determine message type based on current node
        current_node = result.get("current_node", "end")
        market_id = result.get("market_id")
        sports_data = result.get("sports_data")

        if current_node == "wait_for_selection" and market_id:
            # Return market options
            message_type = "market_options"
            data = {
                "market_id": market_id,
                "title": result.get("title", "Prediction Market"),
                "options": result.get("selections", []),
                "bet_amount": result.get("bet_amount", 1.0)
            }
        elif current_node == "wait_for_confirmation":
            # Return confirmation options
            message_type = "confirmation_options"
            data = {
                "market_id": result.get("market_id"),
                "selection": result.get("selected_option"),
                "amount": result.get("bet_amount", 1.0),
                "options": [
                    {"id": "yes", "name": "Yes"},
                    {"id": "no", "name": "No"}
                ]
            }
        elif result.get("status") == "open":
            # Market has been finalized
            message_type = "market_finalized"
            data = {
                "market_id": result.get("market_id"),
                "share_url": f"http://predix/market/{result.get('market_id')}",
                "title": result.get("title"),
                "status": "open"
            }
        elif sports_data:
            # Sports search result
            message_type = "sports_search"
            data = sports_data
        else:
            # Regular text message
            message_type = "text"
            data = None

        return ChatResponse(
            conversation_id=conversation_id,
            message=latest_message,
            message_type=message_type,
            data=data
        )

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return ChatResponse(
            conversation_id=conversation_id,
            message="Sorry, I encountered an error processing your request.",
            message_type="error"
        )

async def process_selection(
        user_id: str,
        market_id: str,
        selection_id: str,
        amount: float,
        conversation_id: str
) -> ChatResponse:
    """
    Process a user's selection using LangGraph.

    Args:
        user_id: ID of the user
        market_id: ID of the market
        selection_id: ID of the selected option
        amount: Bet amount
        conversation_id: ID of the conversation

    Returns:
        ChatResponse with confirmation options
    """
    # Get current state
    current_state = langgraph_store.get_conversation_state(conversation_id)

    if not current_state:
        logger.error(f"Conversation not found: {conversation_id}")
        return ChatResponse(
            conversation_id=conversation_id,
            message="Sorry, I couldn't find that conversation.",
            message_type="error"
        )

    # Update state with selection
    updates = {
        "selected_id": selection_id,
        "bet_amount": amount
    }

    # Update the state
    updated_state = langgraph_store.update_conversation_state(
        conversation_id,
        updates,
        as_node="wait_for_selection"
    )

    # Run the graph to process selection
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": conversation_id}}

    try:
        result = graph.invoke(updated_state, config)

        # Get the last assistant message
        assistant_messages = [msg for msg in result.get("messages", [])
                              if msg["role"] == "assistant"]

        if assistant_messages:
            latest_message = assistant_messages[-1]["content"]
        else:
            latest_message = "I didn't get a response. Please try again."

        # Return confirmation options
        return ChatResponse(
            conversation_id=conversation_id,
            message=latest_message,
            message_type="confirmation_options",
            data={
                "market_id": market_id,
                "selection": result.get("selected_option"),
                "amount": amount,
                "options": [
                    {"id": "yes", "name": "Yes"},
                    {"id": "no", "name": "No"}
                ]
            }
        )

    except Exception as e:
        logger.error(f"Error processing selection: {e}")
        return ChatResponse(
            conversation_id=conversation_id,
            message="Sorry, I encountered an error processing your selection.",
            message_type="error"
        )

async def process_confirmation(
        user_id: str,
        market_id: str,
        confirmed: bool,
        conversation_id: str
) -> ChatResponse:
    """
    Process a user's confirmation using LangGraph.

    Args:
        user_id: ID of the user
        market_id: ID of the market
        confirmed: True if confirmed, False otherwise
        conversation_id: ID of the conversation

    Returns:
        ChatResponse with market finalization details
    """
    # Get current state
    current_state = langgraph_store.get_conversation_state(conversation_id)

    if not current_state:
        logger.error(f"Conversation not found: {conversation_id}")
        return ChatResponse(
            conversation_id=conversation_id,
            message="Sorry, I couldn't find that conversation.",
            message_type="error"
        )

    # Update state with confirmation
    updates = {
        "context": {
            **current_state.get("context", {}),
            "confirmed": confirmed
        }
    }

    # Update the state
    updated_state = langgraph_store.update_conversation_state(
        conversation_id,
        updates,
        as_node="wait_for_confirmation"
    )

    # Run the graph to process confirmation
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": conversation_id}}

    try:
        result = graph.invoke(updated_state, config)

        # Get the last assistant message
        assistant_messages = [msg for msg in result.get("messages", [])
                              if msg["role"] == "assistant"]

        if assistant_messages:
            latest_message = assistant_messages[-1]["content"]
        else:
            latest_message = "I didn't get a response. Please try again."

        if confirmed and result.get("status") == "open":
            # Market has been finalized
            return ChatResponse(
                conversation_id=conversation_id,
                message=latest_message,
                message_type="market_finalized",
                data={
                    "market_id": market_id,
                    "share_url": f"http://predix/market/{market_id}",
                    "title": result.get("title"),
                    "status": "open"
                }
            )
        else:
            # Market creation cancelled
            return ChatResponse(
                conversation_id=conversation_id,
                message=latest_message,
                message_type="text"
            )

    except Exception as e:
        logger.error(f"Error processing confirmation: {e}")
        return ChatResponse(
            conversation_id=conversation_id,
            message="Sorry, I encountered an error processing your confirmation.",
            message_type="error"
        )
