from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.models.chat import Message

"""
Conversation memory management using LangChain's ConversationBufferMemory.
"""

_conversation_memories: dict[str, ConversationBufferMemory] = {}


def get_memory(conversation_id: str) -> ConversationBufferMemory:
    """
    Get or create a ConversationBufferMemory instance for a conversation.

    Args:
        conversation_id: ID of the conversation

    Returns:
        ConversationBufferMemory for the conversation
    """
    if conversation_id not in _conversation_memories:
        _conversation_memories[conversation_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )

    return _conversation_memories[conversation_id]


def get_messages(conversation_id: str) -> list[BaseMessage]:
    """
    Get all messages in a conversation.

    Args:
        conversation_id: ID of the conversation

    Returns:
        List of messages in LangChain format
    """
    memory = get_memory(conversation_id)
    return memory.chat_memory.messages


def get_formatted_messages(conversation_id: str) -> list[Message]:
    """
    Get all messages in a conversation formatted as our Message model.

    Args:
        conversation_id: ID of the conversation

    Returns:
        List of messages in our format
    """
    lc_messages = get_messages(conversation_id)

    messages = []
    for msg in lc_messages:
        if isinstance(msg, HumanMessage):
            messages.append(Message(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(Message(role="assistant", content=msg.content))
        elif isinstance(msg, SystemMessage):
            messages.append(Message(role="system", content=msg.content))

    return messages


def add_user_message(conversation_id: str, message: str) -> None:
    """
    Add a user message to a conversation.

    Args:
        conversation_id: ID of the conversation
        message: Message content
    """
    memory = get_memory(conversation_id)
    memory.chat_memory.add_user_message(message)


def add_ai_message(conversation_id: str, message: str) -> None:
    """
    Add an AI message to a conversation.

    Args:
        conversation_id: ID of the conversation
        message: Message content
    """
    memory = get_memory(conversation_id)
    memory.chat_memory.add_ai_message(message)


def add_system_message(conversation_id: str, message: str) -> None:
    """
    Add a system message to a conversation.

    Args:
        conversation_id: ID of the conversation
        message: Message content
    """
    memory = get_memory(conversation_id)
    memory.chat_memory.add_message(SystemMessage(content=message))


def clear_memory(conversation_id: str) -> None:
    """
    Clear all messages in a conversation.

    Args:
        conversation_id: ID of the conversation
    """
    if conversation_id in _conversation_memories:
        _conversation_memories[conversation_id].clear()


def list_conversations() -> list[str]:
    """
    List all conversation IDs.

    Returns:
        List of conversation IDs
    """
    return list(_conversation_memories.keys())
