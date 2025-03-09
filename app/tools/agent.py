from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.config import logger, settings
from app.models.chat import Message
from app.tools.chat_tools import detect_intent, general_chat_tool, intent_detection_tool
from app.tools.market_tools import create_market_tool, process_selection_tool
from app.tools.sports_tools import sports_info_tool

SYSTEM_PROMPT = """You are an AI assistant for the PrediX prediction market platform.
PrediX allows users to create and participate in prediction markets for sports events.

Your main tasks are:
1. Help users create prediction markets by detecting their intent
2. Answer questions about sports events and prediction markets
3. Provide general assistance

Based on the user's intent, you will take different actions:
- For market creation requests, use the create_prediction_market tool
- For sports information requests, use the get_sports_information tool
- For option selections, use the process_market_selection tool
- For general chat, use the respond_to_general_chat tool

First, always use the detect_intent tool to determine the user's intent.
"""


def initialize_agent(memory: ConversationBufferMemory | None = None):
    """
    Initialize the LangChain agent with tools.

    Args:
        memory: Optional ConversationBufferMemory

    Returns:
        Configured AgentExecutor
    """

    llm = ChatOpenAI(
        model="chatgpt-4o-latest",
        temperature=0.2,
        api_key=settings.OPENAI_API_KEY
    )

    # Define the tools
    tools = [
        intent_detection_tool,
        general_chat_tool,
        sports_info_tool,
        create_market_tool,
        process_selection_tool
    ]

    # Create agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_tool_error=True,
        max_iterations=3,
        early_stopping_method="force",
        memory=memory
    )

    return agent_executor


async def process_agent_message(
        message: str,
        user_id: str,
        conversation_id: str,
        memory: ConversationBufferMemory,
        selected_option: str | None = None,
        bet_amount: float | None = None
) -> dict[str, Any]:
    """
    Process a message through the agent.

    Args:
        message: User message content
        user_id: ID of the user
        conversation_id: ID of the conversation
        memory: ConversationBufferMemory for the conversation
        selected_option: Optional selected market option
        bet_amount: Optional betting amount

    Returns:
        Dictionary with response and relevant data
    """
    logger.info(f"Processing agent message for user {user_id}: {message[:50]}...")

    # Initialize agent with memory
    agent_executor = initialize_agent(memory)

    try:
        # Handle special messages
        if selected_option:
            # Process selection
            result = await process_selection_tool.ainvoke({
                "selected_option": selected_option,
                "bet_amount": bet_amount or 1.0
            })

            # Add AI message to memory
            memory.chat_memory.add_ai_message(result["messages"][0].content)

            return result

        # Detect intent first
        intent = await detect_intent(message)

        if intent == "MARKET_CREATION":
            # Create market
            result = await create_market_tool.ainvoke({
                "message": message,
                "user_id": user_id
            })

            # Add AI message to memory
            memory.chat_memory.add_ai_message(result["messages"][0].content)

            result["intent"] = "MARKET_CREATION"
            return result

        elif intent == "SPORTS_INFO":
            # Get sports info
            result = await sports_info_tool.ainvoke({
                "message": message
            })

            # Add AI message to memory
            memory.chat_memory.add_ai_message(result["messages"][0].content)

            result["intent"] = "SPORTS_INFO"
            return result

        else:
            # Use the agent for general chat
            # The memory is already included, so we can just pass the message
            agent_result = await agent_executor.ainvoke({"input": message})

            result = {
                "messages": [
                    Message(role="assistant", content=agent_result["output"])
                ],
                "intent": "GENERAL_CHAT"
            }

            return result

    except Exception as e:
        logger.error(f"Error processing agent message: {e}")

        # Add error message to memory
        error_message = "I encountered an error processing your request. Please try again."
        memory.chat_memory.add_ai_message(error_message)

        return {
            "messages": [
                Message(role="assistant", content=error_message)
            ],
            "intent": "ERROR"
        }
