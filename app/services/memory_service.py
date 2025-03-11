import json
import logging
import os
from datetime import datetime
from typing import Any

# 메모리 스토어 (인메모리 캐시)
_memory_store: dict[str, list[dict[str, Any]]] = {}

def get_memory_path(conversation_id: str) -> str:
    """
    대화 메모리 파일 경로 가져오기

    Returns:
        파일 경로
    """
    memory_dir = os.path.join(os.getcwd(), "data", "memory")
    os.makedirs(memory_dir, exist_ok=True)
    return os.path.join(memory_dir, f"{conversation_id}.json")

def save_message(conversation_id: str, role: str, content: str) -> None:
    """
    메시지 저장 (메모리와 파일 모두)

    Args:
        conversation_id: 대화 ID
        role: 역할 (user/assistant/system)
        content: 메시지 내용
    """
    if conversation_id not in _memory_store:
        _memory_store[conversation_id] = []

    # 새 메시지 생성
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }

    # 메모리에 추가
    _memory_store[conversation_id].append(message)

    # 파일에 저장
    try:
        file_path = get_memory_path(conversation_id)
        data = {
            "conversation_id": conversation_id,
            "messages": _memory_store[conversation_id],
            "updated_at": datetime.now().isoformat()
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logging.error(f"Error saving message to file: {str(e)}")

def get_memory_messages(conversation_id: str) -> list[dict[str, str]]:
    """
    대화 메시지 가져오기

    Args:
        conversation_id: 대화 ID

    Returns:
        메시지 목록 (role, content 포함)
    """
    # 메모리에서 확인
    if conversation_id in _memory_store:
        return [{"role": msg["role"], "content": msg["content"]}
                for msg in _memory_store[conversation_id]]

    # 파일에서 로드 시도
    file_path = get_memory_path(conversation_id)
    if os.path.exists(file_path):
        try:
            with open(file_path) as f:
                data = json.load(f)

                messages = data.get("messages", [])
                _memory_store[conversation_id] = messages

                return [{"role": msg["role"], "content": msg["content"]}
                        for msg in messages]

        except Exception as e:
            logging.error(f"Error loading messages from file: {str(e)}")

    # 새 대화 생성
    _memory_store[conversation_id] = []
    return []

def get_formatted_messages(conversation_id: str) -> list[dict[str, Any]]:
    """
    포맷팅된 메시지 목록 가져오기 (timestamp 포함)

    Returns:
        메시지 목록 (role, content, timestamp 포함)
    """
    # 메모리에서 가져오기
    messages = get_memory_messages(conversation_id)

    # 타임스탬프 추가
    return [
        {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": (
                _memory_store.get(conversation_id, [])[i].get("timestamp", datetime.now().isoformat())
                if i < len(_memory_store.get(conversation_id, []))
                else datetime.now().isoformat()
            )
        }
        for i, msg in enumerate(messages)
    ]

def list_conversations() -> list[str]:
    """
    모든 대화 ID 목록 가져오기
    """
    memory_dir = os.path.join(os.getcwd(), "data", "memory")
    if not os.path.exists(memory_dir):
        return []

    conversation_files = [f for f in os.listdir(memory_dir) if f.endswith(".json")]
    return [f.replace(".json", "") for f in conversation_files]
