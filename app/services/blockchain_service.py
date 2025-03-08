import uuid
from datetime import datetime
from typing import Any

from app.config import logger
from app.db import langgraph_store


async def create_blockchain_market(market_data: dict[str, Any]) -> dict[str, Any]:
    """
    Create a market on the blockchain via USER BE.
    This is a simulated implementation for PoC.

    Args:
        market_data: Market data to be stored on blockchain

    Returns:
        Response from blockchain operation
    """
    market_id = market_data.get("id", str(uuid.uuid4()))
    user_id = market_data.get("creator_id")

    logger.info(f"Creating blockchain market: {market_id}")

    # Simulate blockchain transaction
    # In production, this would make an HTTP request to USER BE
    tx_hash = f"tx_{uuid.uuid4().hex[:16]}"

    # Store transaction info in user's memory
    if user_id:
        langgraph_store.save_memory(
            user_id,
            "transactions",
            {
                "type": "create_market",
                "market_id": market_id,
                "tx_hash": tx_hash,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
        )

    return {
        "status": "success",
        "market_id": market_id,
        "tx_hash": tx_hash,
        "blockchain": "solana",
        "timestamp": datetime.now().isoformat()
    }


async def place_bet_on_blockchain(
        user_id: str,
        market_id: str,
        selection_id: str,
        amount: float,
        wallet_address: str
) -> dict[str, Any]:
    """
    Place a bet on the blockchain via USER BE.
    This is a simulated implementation for PoC.

    Args:
        user_id: ID of the user
        market_id: ID of the market
        selection_id: ID of the selected option
        amount: Bet amount in SOL
        wallet_address: User's wallet address

    Returns:
        Response from blockchain operation
    """
    logger.info(f"Placing bet on blockchain: user={user_id}, market={market_id}, amount={amount}")

    # Simulate blockchain transaction
    # In production, this would make an HTTP request to USER BE
    tx_hash = f"tx_{uuid.uuid4().hex[:16]}"

    # Store transaction info in user's memory
    langgraph_store.save_memory(
        user_id,
        "transactions",
        {
            "type": "place_bet",
            "market_id": market_id,
            "selection_id": selection_id,
            "amount": amount,
            "tx_hash": tx_hash,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    )

    # Also store in user's bet history
    langgraph_store.save_memory(
        user_id,
        "bets",
        {
            "market_id": market_id,
            "selection_id": selection_id,
            "amount": amount,
            "tx_hash": tx_hash,
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }
    )

    return {
        "status": "success",
        "user_id": user_id,
        "market_id": market_id,
        "selection_id": selection_id,
        "amount": amount,
        "tx_hash": tx_hash,
        "blockchain": "solana",
        "timestamp": datetime.now().isoformat()
    }


async def close_market_on_blockchain(
        market_id: str,
        winning_selection_id: str
) -> dict[str, Any]:
    """
    Close a market on the blockchain via USER BE.
    This is a simulated implementation for PoC.

    Args:
        market_id: ID of the market
        winning_selection_id: ID of the winning selection

    Returns:
        Response from blockchain operation
    """
    logger.info(f"Closing market on blockchain: market={market_id}, winner={winning_selection_id}")

    # Simulate blockchain transaction
    # In production, this would make an HTTP request to USER BE
    tx_hash = f"tx_{uuid.uuid4().hex[:16]}"

    # Get market info to find the creator
    from app.services.market_service import get_market_by_id
    market = get_market_by_id(market_id)

    if market:
        user_id = market.get("creator_id")

        if user_id:
            # Store transaction info in creator's memory
            langgraph_store.save_memory(
                user_id,
                "transactions",
                {
                    "type": "close_market",
                    "market_id": market_id,
                    "winning_selection_id": winning_selection_id,
                    "tx_hash": tx_hash,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
            )

            # Update market status in creator's memory
            from app.services.market_service import update_market_status
            update_market_status(market_id, "closed")

    return {
        "status": "success",
        "market_id": market_id,
        "winning_selection_id": winning_selection_id,
        "tx_hash": tx_hash,
        "blockchain": "solana",
        "timestamp": datetime.now().isoformat()
    }


async def get_user_transactions(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Get a user's blockchain transaction history.

    Args:
        user_id: ID of the user
        limit: Maximum number of transactions to return

    Returns:
        List of transactions
    """
    try:
        memories = langgraph_store.get_memories(user_id, "transactions", limit=limit)

        # Extract transaction data from memories
        transactions = []
        for memory in memories:
            tx_data = memory.get("value", {})
            if "tx_hash" in tx_data:
                transactions.append(tx_data)

        # Sort by timestamp (newest first)
        transactions.sort(key=lambda tx: tx.get("timestamp", ""), reverse=True)

        return transactions

    except Exception as e:
        logger.error(f"Error retrieving transactions for user {user_id}: {e}")
        return []
