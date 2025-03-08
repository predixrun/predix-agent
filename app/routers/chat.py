from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.db import langgraph_store
from app.dependecies import verify_api_key
from app.graph.market_graph import get_compiled_graph
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
    state = langgraph_store.get_conversation_state(conversation_id)

    if not state or "messages" not in state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return {
        "conversation_id": conversation_id,
        "messages": state.get("messages", [])
    }


@router.get("/conversations", response_model=list[str])
async def list_conversations(
        api_key: str = Depends(verify_api_key)
):
    """
    List all conversation IDs.
    This is for debugging purposes in the PoC.
    """
    graph = get_compiled_graph()

    # Get all threads from the checkpointer
    threads = graph.checkpointer.list_threads()

    return list(threads)


@router.get("/conversation_history/{conversation_id}", response_model=list[dict[str, Any]])
async def get_conversation_history(
        conversation_id: str,
        api_key: str = Depends(verify_api_key)
):
    """
    Get the history of state changes for a conversation.
    This is for debugging purposes in the PoC.
    """
    history = langgraph_store.get_conversation_history(conversation_id)

    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation history not found"
        )

    return history


@router.get("/memory/{user_id}/{memory_type}", response_model=list[dict[str, Any]])
async def get_user_memories(
        user_id: str,
        memory_type: str,
        query: str | None = None,
        limit: int = Query(10, ge=1, le=100),
        api_key: str = Depends(verify_api_key)
):
    """
    Get a user's memories of a specific type.
    """
    memories = langgraph_store.get_memories(user_id, memory_type, query, limit)

    if not memories:
        return []

    return memories
