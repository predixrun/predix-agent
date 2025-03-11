import logging
from datetime import datetime
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.models.chat import MessageType

# 시스템 프롬프트
SYSTEM_PROMPT = """
You are an AI assistant for the PrediX prediction market platform. 
PrediX allows users to create and participate in prediction markets for sports(Football) events.
Currently only Football is supported.

Current Date and Time (UTC): {current_datetime}
Current Day: {current_day}

Your main tasks are:
1. Help users create prediction markets for sports events
2. Answer questions about sports events and prediction markets
3. Provide general assistance

When creating a market, you need to collect:
1. Sports event information (teams, league, date) - use search tools to find real events
2. User's prediction option (which team will win vs draw&lose, 현재는 승리 vs 무승부 및 패배 두 그룹으로 나눠진다.)
3. Betting amount (in SOL)

If the user provides incomplete information, ask for clarification.
친구같은 친근한 말투를 사용하라. 
"""

def create_agent():
    """
    ReAct 에이전트 생성 (create_react_agent 사용)
    """

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-4o-2024-11-20",
        temperature=0.2,
        api_key=settings.OPENAI_API_KEY
    )

    # 도구 초기화 (동적 임포트로 순환 참조 방지)
    from app.tools.market_tools import create_market_tool, select_option_tool, set_bet_amount_tool
    from app.tools.sports_tools import fixture_search_tool, league_search_tool, team_search_tool

    tools = [
        league_search_tool,
        team_search_tool,
        fixture_search_tool,
        create_market_tool,
        select_option_tool,
        set_bet_amount_tool
    ]

    # 프롬프트 생성
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_day = datetime.now().strftime("%A")
    prompt = SYSTEM_PROMPT.format(
        current_datetime=current_datetime,
        current_day=current_day
    )

    # create_react_agent 사용하여 에이전트 생성
    # 각 대화별 메모리 사용 (싱글톤 제거)
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=prompt,
        version="v1",
        debug=True
    )

    logging.info("Created new ReAct agent instance")
    return agent

def extract_tool_data(result_state: dict[str, Any]) -> tuple[MessageType, dict[str, Any] | None]:
    """
    도구 실행 결과에서 메시지 타입과 데이터 추출

    Args:
        result_state: 에이전트 실행 결과 상태

    Returns:
        메시지 타입과 데이터 튜플
    """
    message_type = MessageType.TEXT
    data = None

    # 디버깅: 결과 상태 구조 확인
    logging.debug(f"Result state keys: {result_state.keys()}")
    if "intermediate_steps" in result_state:
        logging.debug(f"Found intermediate_steps with {len(result_state['intermediate_steps'])} entries")

    # intermediate_steps가 있는지 확인 (react 에이전트에서는 이 형태로 결과 제공)
    if "intermediate_steps" in result_state:
        for i, step in enumerate(result_state["intermediate_steps"]):
            logging.debug(f"Step {i}: {step}")

            if len(step) < 2:
                continue

            action = step[0]
            observation = step[1]

            # 액션에서 도구 이름 추출
            tool_name = None
            if hasattr(action, "tool"):
                tool_name = action.tool
            elif isinstance(action, dict) and "name" in action:
                tool_name = action["name"]

            if not tool_name:
                continue

            # 도구 결과가 딕셔너리인 경우만 처리
            if not isinstance(observation, dict):
                continue

            # 마켓 생성 도구
            if tool_name == "create_market":
                message_type = MessageType.MARKET_OPTIONS
                data = observation

            # 옵션 선택 도구
            elif tool_name == "select_option":
                message_type = MessageType.BETTING_AMOUNT_REQUEST
                data = observation

            # 베팅 금액 설정 도구
            elif tool_name == "set_bet_amount":
                message_type = MessageType.MARKET_FINALIZED
                data = observation

            # 스포츠 검색 도구
            elif tool_name in ["league_search", "team_search", "fixture_search"]:
                message_type = MessageType.SPORTS_SEARCH
                data = observation.get("sports_data", {})

    return message_type, data

async def process_message(user_id: str, message: str, conversation_id: str, context: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """
    사용자 메시지 처리

    Args:
        user_id: 사용자 ID
        message: 사용자 메시지
        conversation_id: 대화 ID
        context: 이전 대화 컨텍스트 (옵션)

    Returns:
        처리 결과
    """
    from app.services.memory_service import get_memory_messages, save_message

    # 매번 새로운 에이전트 생성 (싱글톤 제거)
    agent = create_agent()

    # 기존 메시지 가져오기
    messages = context or get_memory_messages(conversation_id)

    # 메시지 목록 생성
    messages_list = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    messages_list.append({"role": "user", "content": message})

    # 메모리에 사용자 메시지 저장
    save_message(conversation_id, "user", message)

    try:
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }

        logging.info(f"Invoking agent with message: {message}")

        # 에이전트 실행
        result = await agent.ainvoke(
            {"messages": messages_list},
            config=config
        )

        logging.debug(f"Agent result keys: {result.keys()}")

        # 결과 처리
        final_message = result["messages"][-1] if "messages" in result else None
        final_content = final_message.content if final_message else "I couldn't process your request."

        # 메모리에 응답 저장
        save_message(conversation_id, "assistant", final_content)

        # 도구 실행 결과에서 메시지 타입과 데이터 추출
        message_type, data = extract_tool_data(result)

        # 응답 생성
        return {
            "conversation_id": conversation_id,
            "message": final_content,
            "message_type": message_type,
            "data": data
        }

    except Exception as e:
        logging.error(f"Error processing message: {str(e)}", exc_info=True)

        # 에러 메시지 추가
        error_message = "Sorry, I encountered an error processing your request. Please try again."
        save_message(conversation_id, "assistant", error_message)

        return {
            "conversation_id": conversation_id,
            "message": error_message,
            "message_type": MessageType.ERROR,
            "data": None
        }
