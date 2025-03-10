import uuid

from app.config import logger
from app.models.chat import ChatResponse, MessageType
from app.services.memory_service import add_ai_message, add_user_message, get_memory
from app.tools.agent import process_agent_message


async def process_chat_message(
    user_id: str,
    message: str,
    conversation_id: str | None = None
) -> ChatResponse:
    """
    Process a chat message from a user.

    Args:
        user_id: ID of the user
        message: The message content
        conversation_id: Optional ID of the conversation

    Returns:
        ChatResponse object with appropriate response
    """
    # Generate conversation ID if not provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    # Log the incoming message
    logger.info(f"Received message from user {user_id} in conversation {conversation_id}")

    # Get memory for this conversation
    memory = get_memory(conversation_id)

    # Check if message is a selection or bet amount
    selected_option = None
    bet_amount = None
    original_message = message

    if message.startswith("SELECTION:"):
        try:
            # Format: SELECTION:option_name
            selected_option = message[10:].strip()
            # Don't add selection commands to memory
            message = f"I select the option: {selected_option}"
        except Exception as e:
            logger.error(f"Error processing selection: {e}")
    elif message.startswith("BET_AMOUNT:"):
        try:
            # Format: BET_AMOUNT:1.5
            bet_amount = float(message[11:].strip())
            # Don't add bet amount commands to memory
            message = f"I want to bet {bet_amount} SOL"
        except Exception as e:
            logger.error(f"Error processing bet amount: {e}")

    # Add user message to memory
    add_user_message(conversation_id, message)

    try:
        # Process the message through the agent
        result = await process_agent_message(
            message=message,
            user_id=user_id,
            conversation_id=conversation_id,
            memory=memory,
            selected_option=selected_option,
            bet_amount=bet_amount
        )

        # Get the assistant message from the result
        assistant_messages = result.get("messages", [])

        if assistant_messages:
            latest_message = assistant_messages[0].content
            # The process_agent_message function already adds AI messages to memory
        else:
            latest_message = "I didn't get a response. Please try again."
            # Add to conversation memory
            add_ai_message(conversation_id, latest_message)

        latest_message.replace("*", "")

        # Determine message type based on intent and context
        intent = result.get("intent", "GENERAL_CHAT")
        context = result.get("context", {})
        market_package = context.get("market_package", {})
        sports_data = result.get("sports_data")

        if intent == "MARKET_CREATION" and market_package:
            # Return market options
            message_type = MessageType.MARKET_OPTIONS
            data = market_package
        elif selected_option:
            # Return betting amount request
            message_type = MessageType.BETTING_AMOUNT_REQUEST
            data = {
                "selected_option": selected_option,
                "initial_amount": bet_amount or result.get("bet_amount", 1.0)
            }
        elif sports_data:
            # Sports search result
            message_type = MessageType.SPORTS_SEARCH
            data = sports_data
        else:
            # Regular text message
            message_type = MessageType.TEXT
            data = None

        return ChatResponse(
            conversation_id=conversation_id,
            message=latest_message,
            message_type=message_type,
            data=data
        )

    except Exception as e:
        logger.error(f"Error processing message: {e}")

        # Add error message to memory
        error_message = "Sorry, I encountered an error processing your request."
        add_ai_message(conversation_id, error_message)

        return ChatResponse(
            conversation_id=conversation_id,
            message=error_message,
            message_type=MessageType.ERROR
        )
