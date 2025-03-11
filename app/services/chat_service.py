import logging
import uuid

from fastapi import HTTPException

from app.agent import process_message
from app.models.chat import ChatResponse, MessageType


async def process_chat_message(
        user_id: str,
        message: str,
        conversation_id: str | None = None
) -> ChatResponse:
    """
    사용자 채팅 메시지 처리

    Args:
        user_id: 사용자 ID
        message: 메시지 내용
        conversation_id: 대화 ID (없으면 생성)

    Returns:
        ChatResponse 객체
    """
    # 대화 ID 없으면 오류처리
    if not conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID is required")

    # logging.info(f"Processing chat message from user {user_id} in conversation {conversation_id}")

    try:
        # 에이전트 처리
        result = await process_message(user_id, message, conversation_id)

        # 응답 생성
        return ChatResponse(
            conversation_id=result["conversation_id"],
            message=result["message"],
            message_type=result["message_type"],
            data=result["data"]
        )

    except Exception as e:
        logging.error(f"Error processing chat message: {str(e)}")

        # 에러 응답 생성
        error_message = "I encountered an error processing your request. Please try again."

        return ChatResponse(
            conversation_id=conversation_id,
            message=error_message,
            message_type=MessageType.ERROR,
            data=None
        )
