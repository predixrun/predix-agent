from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from app.dependecies import verify_api_key
from app.services.blockchain_service import (
    close_market_on_blockchain,
    create_blockchain_market,
    get_user_transactions,
    place_bet_on_blockchain,
)

router = APIRouter()


@router.post("/blockchain/create_market", response_model=dict[str, Any])
async def create_market(
        market_data: dict[str, Any] = Body(...),
        api_key: str = Depends(verify_api_key)
):
    """
    Create a market on the blockchain.
    This endpoint is called by USER BE.
    """
    return await create_blockchain_market(market_data)


@router.post("/blockchain/place_bet", response_model=dict[str, Any])
async def place_bet(
        bet_data: dict[str, Any] = Body(...),
        api_key: str = Depends(verify_api_key)
):
    """
    Place a bet on the blockchain.
    This endpoint is called by USER BE.
    """
    return await place_bet_on_blockchain(
        user_id=bet_data.get("user_id"),
        market_id=bet_data.get("market_id"),
        selection_id=bet_data.get("selection_id"),
        amount=bet_data.get("amount"),
        wallet_address=bet_data.get("wallet_address")
    )


@router.post("/blockchain/close_market", response_model=dict[str, Any])
async def close_market(
        close_data: dict[str, Any] = Body(...),
        api_key: str = Depends(verify_api_key)
):
    """
    Close a market on the blockchain.
    This endpoint is called by USER BE or an automated service.
    """
    return await close_market_on_blockchain(
        market_id=close_data.get("market_id"),
        winning_selection_id=close_data.get("winning_selection_id")
    )


@router.get("/blockchain/transactions/{user_id}", response_model=list[dict[str, Any]])
async def get_transactions(
        user_id: str,
        limit: int = Query(10, ge=1, le=100),
        api_key: str = Depends(verify_api_key)
):
    """
    Get a user's blockchain transaction history.
    """
    return await get_user_transactions(user_id, limit)
