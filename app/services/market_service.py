import logging
from typing import Any

from app.db import langgraph_store

logger = logging.getLogger(__name__)


def get_market_by_id(market_id: str) -> dict[str, Any] | None:
    """
    Get a market by its ID.

    In this LangGraph implementation, we retrieve market data from user memories
    since markets are stored as memories in the LangGraph store.

    Args:
        market_id: ID of the market

    Returns:
        Market data or None if not found
    """
    # Try to find the market in any user's memories
    try:
        # Get all memory namespaces that contain markets
        # This is simplified - in production you would have a more efficient lookup
        market_memories = []

        # This is inefficient but is a simple approach for PoC
        # In production, you would have a database index
        all_users = langgraph_store.store.list_namespaces()

        for user_namespace in all_users:
            if len(user_namespace) >= 2 and user_namespace[1] == "markets":
                user_id = user_namespace[0]
                user_markets = langgraph_store.get_memories(user_id, "markets")
                market_memories.extend(user_markets)

        # Find the market with matching ID
        for memory in market_memories:
            market_data = memory.get("value", {})
            if market_data.get("market_id") == market_id:
                return market_data

        return None

    except Exception as e:
        logger.error(f"Error retrieving market {market_id}: {e}")
        return None


def get_user_markets(user_id: str) -> list[dict[str, Any]]:
    """
    Get all markets created by a user.

    Args:
        user_id: ID of the user

    Returns:
        List of markets created by the user
    """
    try:
        memories = langgraph_store.get_memories(user_id, "markets")

        # Extract market data from memories
        markets = []
        for memory in memories:
            market_data = memory.get("value", {})
            if "market_id" in market_data:
                markets.append(market_data)

        return markets

    except Exception as e:
        logger.error(f"Error retrieving markets for user {user_id}: {e}")
        return []


def get_trending_markets(limit: int = 10) -> list[dict[str, Any]]:
    """
    Get trending markets.
    This is a simplified implementation that returns recent markets.

    Args:
        limit: Maximum number of markets to return

    Returns:
        List of trending markets
    """
    try:
        # Get all markets from all users
        all_markets = []

        # This is inefficient but is a simple approach for PoC
        # In production, you would have a database index
        all_users = langgraph_store.store.list_namespaces()

        for user_namespace in all_users:
            if len(user_namespace) >= 2 and user_namespace[1] == "markets":
                user_id = user_namespace[0]
                user_markets = langgraph_store.get_memories(user_id, "markets")
                all_markets.extend([m.get("value", {}) for m in user_markets])

        # Sort by creation time (newest first)
        all_markets.sort(key=lambda m: m.get("created_at", ""), reverse=True)

        # Return limited number
        return all_markets[:limit]

    except Exception as e:
        logger.error(f"Error retrieving trending markets: {e}")
        return []


def update_market_status(market_id: str, status: str) -> bool:
    """
    Update the status of a market.

    Args:
        market_id: ID of the market
        status: New status

    Returns:
        True if successful, False otherwise
    """
    try:
        # Find the market
        market = get_market_by_id(market_id)

        if not market:
            logger.error(f"Market not found: {market_id}")
            return False

        # Get the user who created the market
        user_id = market.get("creator_id")

        if not user_id:
            logger.error(f"Creator not found for market: {market_id}")
            return False

        # Update the market status
        updated_market = {**market, "status": status}

        # Save the updated market
        langgraph_store.save_memory(
            user_id,
            "markets",
            updated_market,
            memory_id=market_id
        )

        return True

    except Exception as e:
        logger.error(f"Error updating market {market_id}: {e}")
        return False


def find_conversation_for_market(market_id: str) -> str | None:
    """
    Find the conversation ID that created a market.

    Args:
        market_id: ID of the market

    Returns:
        Conversation ID or None if not found
    """
    try:
        # This is a naive implementation that searches all threads
        # In production, you would have a more efficient lookup

        # Get compiled graph to access thread history
        from app.graph.market_graph import get_compiled_graph
        graph = get_compiled_graph()

        # Get all threads
        all_threads = graph.checkpointer.list_threads()

        for thread_id in all_threads:
            # Get state for this thread
            state = langgraph_store.get_conversation_state(thread_id)

            # Check if this thread created the market
            if state and state.get("market_id") == market_id:
                return thread_id

        return None

    except Exception as e:
        logger.error(f"Error finding conversation for market {market_id}: {e}")
        return None
