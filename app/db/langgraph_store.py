from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from app.config import logger

# Create a memory saver for checkpointing
checkpointer = MemorySaver()

# Create an in-memory store for cross-thread data
store = InMemoryStore()


def get_conversation_state(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    """
    Get the current state of a conversation from LangGraph checkpoints.

    Args:
        thread_id: The thread ID of the conversation
        checkpoint_id: Optional checkpoint ID to retrieve a specific state

    Returns:
        The current state of the conversation
    """
    from app.graph.market_graph import get_compiled_graph

    graph = get_compiled_graph()

    config = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id

    try:
        state_snapshot = graph.get_state(config)
        return state_snapshot.values
    except Exception as e:
        logger.error(f"Error retrieving state for thread {thread_id}: {e}")
        return {}


def get_conversation_history(thread_id: str) -> list[dict[str, Any]]:
    """
    Get the history of a conversation from LangGraph checkpoints.

    Args:
        thread_id: The thread ID of the conversation

    Returns:
        A list of state snapshots representing the conversation history
    """
    from app.graph.market_graph import get_compiled_graph

    graph = get_compiled_graph()

    config = {"configurable": {"thread_id": thread_id}}

    try:
        history = list(graph.get_state_history(config))
        return [snapshot.values for snapshot in history]
    except Exception as e:
        logger.error(f"Error retrieving history for thread {thread_id}: {e}")
        return []


def update_conversation_state(thread_id: str, values: dict[str, Any],
                              checkpoint_id: str | None = None,
                              as_node: str | None = None) -> dict[str, Any]:
    """
    Update the state of a conversation in LangGraph checkpoints.

    Args:
        thread_id: The thread ID of the conversation
        values: The values to update in the state
        checkpoint_id: Optional checkpoint ID to update from a specific state
        as_node: Optional node name to attribute the update to

    Returns:
        The updated state
    """
    from app.graph.market_graph import get_compiled_graph

    graph = get_compiled_graph()

    config = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id

    try:
        updated_state = graph.update_state(config, values, as_node=as_node)
        return updated_state.values
    except Exception as e:
        logger.error(f"Error updating state for thread {thread_id}: {e}")
        return {}


def save_memory(user_id: str, memory_type: str, memory_data: dict[str, Any], memory_id: str | None = None) -> str:
    """
    Save a memory to the store for cross-thread access.

    Args:
        user_id: The user ID to associate the memory with
        memory_type: The type of memory (e.g., "preferences", "markets")
        memory_data: The memory data to save
        memory_id: Optional memory ID to use

    Returns:
        The memory ID
    """
    import uuid

    if not memory_id:
        memory_id = str(uuid.uuid4())

    namespace = (user_id, memory_type)

    try:
        store.put(namespace, memory_id, memory_data)
        return memory_id
    except Exception as e:
        logger.error(f"Error saving memory for user {user_id}: {e}")
        return ""


def get_memories(user_id: str, memory_type: str, query: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """
    Retrieve memories from the store.

    Args:
        user_id: The user ID to retrieve memories for
        memory_type: The type of memory to retrieve
        query: Optional query to search for specific memories
        limit: Maximum number of memories to retrieve

    Returns:
        A list of memory items
    """
    namespace = (user_id, memory_type)

    try:
        if query:
            memories = store.search(namespace, query=query, limit=limit)
        else:
            memories = store.search(namespace, limit=limit)

        return [memory.dict() for memory in memories]
    except Exception as e:
        logger.error(f"Error retrieving memories for user {user_id}: {e}")
        return []


def delete_memory(user_id: str, memory_type: str, memory_id: str) -> bool:
    """
    Delete a memory from the store.

    Args:
        user_id: The user ID associated with the memory
        memory_type: The type of memory
        memory_id: The ID of the memory to delete

    Returns:
        True if successful, False otherwise
    """
    namespace = (user_id, memory_type)

    try:
        store.delete(namespace, memory_id)
        return True
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id} for user {user_id}: {e}")
        return False
