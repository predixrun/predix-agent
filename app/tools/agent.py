from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.config import logger, settings
from app.models.chat import Message
from app.tools.chat_tools import general_chat_tool
from app.tools.market_tools import create_market_tool, process_selection_tool
from app.tools.sports_tools import sports_info_tool

SYSTEM_PROMPT = """You are an AI assistant for the PrediX prediction market platform.
PrediX allows users to create and participate in prediction markets for sports events.

Based on the user's message, choose the appropriate tool:
- If the user wants to create a prediction market (mentions creating a market, betting on a team, etc.), use the create_prediction_market tool
- If the user asks about sports information (teams, matches, scores, etc.), use the get_sports_information tool
- If the user selects a market option or provides a betting amount, use the process_market_selection tool
- For general questions and conversations, use the respond_to_general_chat tool

directly select the appropriate tool based on the user's message content.
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
        model="gpt-4o-2024-11-20",
        temperature=0.1,
        api_key=settings.OPENAI_API_KEY
    )

    # Define the tools
    tools = [
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
        # Handle special messages for market selections directly
        if selected_option:
            # Process selection without going through the agent
            result = await process_selection_tool.ainvoke({
                "selected_option": selected_option,
                "bet_amount": bet_amount or 1.0
            })

            # Add AI message to memory
            memory.chat_memory.add_ai_message(result["messages"][0].content)

            return result

        # Use the agent for all other messages
        agent_result = await agent_executor.ainvoke({"input": message})

        # Extract intent information from the tool used
        intent = "GENERAL_CHAT"
        if agent_result.get("intermediate_steps"):
            tool_name = agent_result["intermediate_steps"][0][0].tool
            if tool_name == "create_prediction_market":
                intent = "MARKET_CREATION"
            elif tool_name == "get_sports_information":
                intent = "SPORTS_INFO"

        # Extract additional context from tool outputs
        context = {}
        sports_data = None

        for step in agent_result.get("intermediate_steps", []):
            tool_output = step[1]
            if isinstance(tool_output, dict):
                if "market_package" in tool_output.get("context", {}):
                    context = tool_output.get("context", {})
                if "sports_data" in tool_output:
                    sports_data = tool_output.get("sports_data")

        result = {
            "messages": [
                Message(role="assistant", content=agent_result["output"])
            ],
            "intent": intent,
            "context": context
        }

        if sports_data:
            result["sports_data"] = sports_data

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
