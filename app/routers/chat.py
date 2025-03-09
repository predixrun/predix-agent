from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependecies import verify_api_key
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import process_chat_message

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
        request: ChatRequest,
        api_key: str = Depends(verify_api_key)
):
    """
    Process a chat message from a user.
    """
    return await process_chat_message(
        user_id=request.user_id,
        message=request.message,
        conversation_id=request.conversation_id,
    )


@router.get("/conversations/{conversation_id}", response_model=dict[str, Any])
async def get_conversation(
        conversation_id: str,
        api_key: str = Depends(verify_api_key)
):
    """
    Get all messages in a conversation.
    """
    from app.services.memory_service import get_formatted_messages

    messages = get_formatted_messages(conversation_id)

    if not messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return {
        "conversation_id": conversation_id,
        "messages": messages
    }


@router.get("/conversations", response_model=list[str])
async def list_all_conversations(
        api_key: str = Depends(verify_api_key)
):
    """
    List all conversation IDs.
    For PoC debugging.

    """
    from app.services.memory_service import list_conversations

    return list_conversations()
