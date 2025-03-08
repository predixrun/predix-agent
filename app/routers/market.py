from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependecies import verify_api_key
from app.models.chat import ChatResponse
from app.models.market import ConfirmationRequest, SelectionRequest
from app.services.chat_service import process_confirmation, process_selection
from app.services.market_service import find_conversation_for_market, get_market_by_id, get_trending_markets, get_user_markets

router = APIRouter()


@router.post("/selection", response_model=ChatResponse)
async def selection(
        request: SelectionRequest,
        api_key: str = Depends(verify_api_key)
):
    """
    Process a user's selection of a market option.
    """
    return await process_selection(
        user_id=request.user_id,
        market_id=request.market_id,
        selection_id=request.selection_id,
        amount=request.amount,
        conversation_id=request.conversation_id
    )


@router.post("/confirm", response_model=ChatResponse)
async def confirm(
        request: ConfirmationRequest,
        api_key: str = Depends(verify_api_key)
):
    """
    Process a user's confirmation to create a market.
    """
    return await process_confirmation(
        user_id=request.user_id,
        market_id=request.market_id,
        confirmed=request.confirmed,
        conversation_id=request.conversation_id
    )


@router.get("/markets/{market_id}", response_model=dict[str, Any])
async def get_market(
        market_id: str,
        api_key: str = Depends(verify_api_key)
):
    """
    Get a market by ID.
    """
    market = get_market_by_id(market_id)

    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )

    return market


@router.get("/markets/user/{user_id}", response_model=list[dict[str, Any]])
async def get_markets_by_user(
        user_id: str,
        api_key: str = Depends(verify_api_key)
):
    """
    Get markets created by a user.
    """
    markets = get_user_markets(user_id)
    return markets


@router.get("/markets/trending", response_model=list[dict[str, Any]])
async def get_trending_markets_endpoint(
        limit: int = Query(10, ge=1, le=100),
        api_key: str = Depends(verify_api_key)
):
    """
    Get trending markets.
    """
    markets = get_trending_markets(limit)
    return markets


@router.get("/markets/conversation/{market_id}", response_model=dict[str, Any])
async def get_market_conversation(
        market_id: str,
        api_key: str = Depends(verify_api_key)
):
    """
    Get the conversation that created a market.
    """
    conversation_id = find_conversation_for_market(market_id)

    if not conversation_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found for market"
        )

    return {"market_id": market_id, "conversation_id": conversation_id}
