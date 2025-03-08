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
        current_state = langgraph_store.get_conversation_state(conversation_id)

        # If we have no state or empty state, initialize it
        if not current_state:
            current_state = {
                "messages": [],
                "title": None,
                "description": None,
                "status": None,
                "selections": [],
                "selected_option": None,
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
            "title": None,
            "description": None,
            "status": None,
            "selections": [],
            "selected_option": None,
            "bet_amount": None,
            "creator_id": user_id,
            "sports_data": None,
            "current_node": None,
            "context": {"user_id": user_id, "wallet_address": wallet_address}
        }

    # Add user message to conversation
    user_message = {"role": "user", "content": message}

    # Check if message is a selection
    if message.startswith("SELECTION:"):
        try:
            # Format: SELECTION:option_name
            selected_option = message[10:].strip()
            current_state["selected_option"] = selected_option
            # Skip adding this technical message to the conversation history
        except Exception as e:
            logger.error(f"Error processing selection: {e}")
            # Add as normal message if parsing fails
            if "messages" in current_state:
                messages = current_state["messages"] + [user_message]
            else:
                messages = [user_message]
            current_state["messages"] = messages
    elif message.startswith("BET_AMOUNT:"):
        try:
            # Format: BET_AMOUNT:1.5
            bet_amount = float(message[11:].strip())
            current_state["bet_amount"] = bet_amount
            # Skip adding this technical message to the conversation history
        except Exception as e:
            logger.error(f"Error processing bet amount: {e}")
            # Add as normal message if parsing fails
            if "messages" in current_state:
                messages = current_state["messages"] + [user_message]
            else:
                messages = [user_message]
            current_state["messages"] = messages
    else:
        # Add as normal message
        if "messages" in current_state:
            current_state["messages"] = current_state["messages"] + [user_message]
        else:
            current_state["messages"] = [user_message]

    # Run the graph
    config = {"configurable": {"thread_id": conversation_id}}

    try:
        result = await graph.ainvoke(current_state, config)

        # Get the last assistant message
        assistant_messages = [msg for msg in result.get("messages", [])
                            if msg["role"] == "assistant"]

        if assistant_messages:
            latest_message = assistant_messages[-1]["content"]
        else:
            latest_message = "I didn't get a response. Please try again."

        # Determine message type based on current node and context
        current_node = result.get("current_node", "end")
        sports_data = result.get("sports_data")
        context = result.get("context", {})
        market_package = context.get("market_package", {})

        if current_node == "market_options" and market_package:
            # Return market options
            message_type = "market_options"
            data = market_package
        elif message.startswith("SELECTION:") and result.get("selected_option"):
            # Return betting amount request
            message_type = "betting_amount_request"
            data = {
                "selected_option": result.get("selected_option"),
                "initial_amount": result.get("bet_amount", 1.0)
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
