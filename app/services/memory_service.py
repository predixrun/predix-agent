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
        "type": "text",
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

def save_tool_message(conversation_id: str, tool_call_id: str, content: str,
                      status: str = "success", artifact: Any | None = None) -> None:
    """
    도구 메시지 저장 (메모리와 파일 모두)
    모든 메시지가 _memory_store에 통합되어 저장됨

    Args:
        conversation_id: 대화 ID
        tool_call_id: 도구 호출 ID
        content: 메시지 내용
        status: 도구 실행 상태 ('success' 또는 'error')
        artifact: 추가 데이터
    """
    if conversation_id not in _memory_store:
        _memory_store[conversation_id] = []

    # 새 도구 메시지 생성
    tool_message = {
        "type": "tool",
        "role": "tool",  # 역할도 추가하여 일관성 유지
        "tool_call_id": tool_call_id,
        "content": content,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

    # artifact가 있으면 추가
    if artifact:
        tool_message["artifact"] = artifact

    # 메모리에 추가
    _memory_store[conversation_id].append(tool_message)

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
        logging.error(f"Error saving tool message to file: {str(e)}")

def get_memory_messages(conversation_id: str) -> list[dict[str, str]]:
    """
    일반 대화 메시지만 가져오기 (tool 타입 제외)

    Args:
        conversation_id: 대화 ID

    Returns:
        메시지 목록 (role, content 포함)
    """
    # 메모리에서 가져와 필터링
    if conversation_id in _memory_store:
        return [{"role": msg["role"], "content": msg["content"]}
                for msg in _memory_store[conversation_id]
                if msg.get("type") != "tool"]  # tool 타입 제외

    # 파일에서 로드 시도
    file_path = get_memory_path(conversation_id)
    if os.path.exists(file_path):
        try:
            with open(file_path) as f:
                data = json.load(f)

                messages = data.get("messages", [])
                _memory_store[conversation_id] = messages

                # 일반 메시지만 필터링하여 반환
                return [{"role": msg["role"], "content": msg["content"]}
                        for msg in messages
                        if msg.get("type") != "tool"]

        except Exception as e:
            logging.error(f"Error loading messages from file: {str(e)}")

    # 새 대화 생성
    _memory_store[conversation_id] = []
    return []

def get_tool_messages(conversation_id: str) -> list[dict[str, Any]]:
    """
    도구 메시지만 가져오기

    Args:
        conversation_id: 대화 ID

    Returns:
        도구 메시지 목록
    """
    # 메모리에서 도구 메시지만 필터링
    if conversation_id in _memory_store:
        return [msg for msg in _memory_store[conversation_id]
                if msg.get("type") == "tool"]

    # 파일에서 로드 시도
    file_path = get_memory_path(conversation_id)
    if os.path.exists(file_path):
        try:
            with open(file_path) as f:
                data = json.load(f)

                messages = data.get("messages", [])
                _memory_store[conversation_id] = messages

                # 도구 메시지만 필터링하여 반환
                return [msg for msg in messages
                        if msg.get("type") == "tool"]

        except Exception as e:
            logging.error(f"Error loading messages from file: {str(e)}")

    # 메시지가 없는 경우
    return []

def get_formatted_messages(conversation_id: str) -> list[dict[str, Any]]:
    """
    포맷팅된 일반 메시지 목록 가져오기 (timestamp 포함, tool 타입 제외)

    Returns:
        메시지 목록 (role, content, timestamp 포함)
    """
    # 메모리에서 가져오기
    if conversation_id not in _memory_store:
        # 파일에서 로드 시도
        file_path = get_memory_path(conversation_id)
        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    _memory_store[conversation_id] = data.get("messages", [])
            except Exception as e:
                logging.error(f"Error loading messages from file: {str(e)}")
                _memory_store[conversation_id] = []
        else:
            _memory_store[conversation_id] = []

    # 일반 메시지만 필터링
    return [
        {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg.get("timestamp", datetime.now().isoformat())
        }
        for msg in _memory_store[conversation_id]
        if msg.get("type") != "tool"
    ]

def get_all_messages(conversation_id: str) -> dict[str, Any]:
    """
    모든 메시지 가져오기 (일반 메시지와 도구 메시지 분리)

    Args:
        conversation_id: 대화 ID

    Returns:
        일반 메시지와 도구 메시지를 포함한 딕셔너리
    """
    return {
        "conversation_id": conversation_id,
        "messages": get_formatted_messages(conversation_id),
        "tool_messages": get_tool_messages(conversation_id)
    }

def get_raw_messages(conversation_id: str) -> list[dict[str, Any]]:
    """
    모든 메시지를 순서대로 가져오기 (타입 구분 없이 원본 형태 그대로)

    Args:
        conversation_id: 대화 ID

    Returns:
        모든 메시지 목록
    """
    if conversation_id in _memory_store:
        return _memory_store[conversation_id]

    # 파일에서 로드 시도
    file_path = get_memory_path(conversation_id)
    if os.path.exists(file_path):
        try:
            with open(file_path) as f:
                data = json.load(f)
                messages = data.get("messages", [])
                _memory_store[conversation_id] = messages
                return messages
        except Exception as e:
            logging.error(f"Error loading messages from file: {str(e)}")

    # 메시지가 없는 경우
    _memory_store[conversation_id] = []
    return []

def list_conversations() -> list[str]:
    """
    모든 대화 ID 목록 가져오기
    """
    memory_dir = os.path.join(os.getcwd(), "data", "memory")
    if not os.path.exists(memory_dir):
        return []

    conversation_files = [f for f in os.listdir(memory_dir) if f.endswith(".json")]
    return [f.replace(".json", "") for f in conversation_files]
