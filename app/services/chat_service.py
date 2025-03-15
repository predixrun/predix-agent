import logging

from fastapi import HTTPException

from app.agent import process_message
from app.models.chat import ChatResponse, MessageType
from app.services.memory_service import get_tool_messages


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
        result = await process_message(message, conversation_id)

        # 도구 메시지 가져오기
        tool_messages = get_tool_messages(conversation_id)

        # 도구 메시지가 있고 결과에 데이터가 없는 경우 가장 최근 도구 메시지의 아티팩트를 사용
        if tool_messages and not result["data"]:
            # 가장 최근 도구 메시지 데이터를 사용
            latest_tool_message = tool_messages[-1]
            if "artifact" in latest_tool_message:
                result["data"] = latest_tool_message["artifact"]

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
