from fastapi import APIRouter, Depends

from app.dependecies import verify_api_key
from app.models.chat import ChatResponse
from app.models.market import ConfirmationRequest, SelectionRequest
from app.services.chat_service import process_confirmation, process_selection

router = APIRouter()


@router.post("/selection", response_model=ChatResponse)
async def selection(
        request: SelectionRequest,
        api_key: str = Depends(verify_api_key)
):
    """
    미사용 Process a user's selection of a market option.
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
    미사용 Process a user's confirmation to create a market.
    """
    return await process_confirmation(
        user_id=request.user_id,
        market_id=request.market_id,
        confirmed=request.confirmed,
        conversation_id=request.conversation_id
    )
