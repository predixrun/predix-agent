from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependecies import verify_api_key
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import process_chat_message
from app.services.memory_service import get_formatted_messages, list_conversations

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(
        request: ChatRequest,
        api_key: str = Depends(verify_api_key)
):
    """
    채팅 메시지 처리
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
    대화 내역 조회

    Returns:
        대화 메시지 목록
    """
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
    모든 대화 ID 목록 조회
    """
    return list_conversations()
