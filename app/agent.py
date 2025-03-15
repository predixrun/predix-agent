import logging
from datetime import datetime
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.models.chat import MessageType
from app.services.memory_service import save_tool_message

# 시스템 프롬프트
SYSTEM_PROMPT = """
You are an AI assistant for the PrediX prediction market platform. 
PrediX allows users to create and participate in prediction markets for sports(Football) events.
Currently only Football is supported.

Current Date (UTC): {current_datetime}, {current_day}

Your main tasks are:
1. Help users create prediction markets for sports events
2. Answer questions about sports events and prediction markets

Your role is to gather information and prepare data for display.
You DO NOT directly interact with blockchain or create actual markets - that's handled by a separate backend service.
Your tools format data that will be shown to users as cards or buttons in the frontend.

When helping users create a market, you need to collect:
1. Sports event information (teams, date) - use search tools to find real events. fixture_id도 괄호로 감싸고 알려주세요.
2. User's prediction option (which team will win vs draw&lose, 현재는 승리 vs 무승부 및 패배 두 그룹으로 나눠진다.)
3. Betting amount (in SOL) 반드시 유저에게 얼마를 베팅할 것인지 물어봐야 한다.

 유저에게 선택 옵션 혹은 베팅 금액 제시, 마켓생성을 할때, 아래의 툴을 반드시 선택해야 합니다.
- dp_asking_options
- set_bet_amount_dp_tool
- create_market_dp_tool
It will display appropriate FE UI elements to the user, which btn communicates actual blk server.

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
    from app.tools import create_market_dp_tool, dp_asking_options, set_bet_amount_dp_tool
    from app.tools.sports_tools import fixture_search_tool, league_search_tool, team_search_tool

    tools = [
        league_search_tool,
        team_search_tool,
        fixture_search_tool,
        create_market_dp_tool,
        dp_asking_options,
        set_bet_amount_dp_tool
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
    import json

    message_type = MessageType.TEXT
    data = None

    # 디버깅: 결과 상태 구조 확인
    logging.debug(f"Result state keys: {result_state.keys()}")

    # messages 배열에서 ToolMessage 찾기
    if "messages" in result_state:
        from langchain_core.messages import ToolMessage

        for msg in result_state["messages"]:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None)
                tool_call_id = getattr(msg, "tool_call_id", None)

                if not tool_name:
                    continue

                # 디버깅: 도구 호출 로깅
                logging.debug(f"Tool called: {tool_name}")

                # ToolMessage에서 content 추출
                content = msg.content

                # content가 문자열이면 JSON 파싱 시도
                if isinstance(content, str):
                    try:
                        content_data = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        content_data = {"message": str(content)}
                else:
                    content_data = content

                # 도구 메시지 저장 (호출 ID가 없는 경우 생성)
                if tool_call_id is None:
                    tool_call_id = f"call_{tool_name}_{datetime.now().timestamp()}"

                # 도구 메시지 저장
                save_tool_message(
                    conversation_id=result_state.get("configurable", {}).get("thread_id", "unknown"),
                    tool_call_id=tool_call_id,
                    content=str(content),
                    status="success",
                    artifact=content_data,
                )

                # 도구 유형에 따라 메시지 타입과 데이터 설정
                if tool_name == "dp_asking_options":
                    message_type = MessageType.MARKET_OPTIONS
                    data = content_data
                    logging.info(f"Market options created: {tool_name}")

                elif tool_name == "set_bet_amount_dp_tool":
                    message_type = MessageType.BETTING_AMOUNT_REQUEST
                    data = content_data
                    logging.info(f"Betting amount requested: {tool_name}")

                elif tool_name == "create_market_dp_tool":
                    message_type = MessageType.MARKET_FINALIZED
                    data = content_data
                    logging.info(f"Market finalized: {tool_name}")

                elif tool_name in ["league_search", "team_search", "fixture_search"]:
                    message_type = MessageType.SPORTS_SEARCH
                    # 스포츠 데이터가 있으면 사용, 없으면 전체 content 사용
                    if "sports_data" in content_data:
                        data = content_data["sports_data"]
                    elif "fixtures" in content_data or "teams" in content_data or "leagues" in content_data:
                        data = content_data
                    else:
                        data = {"message": content_data.get("message", "Sports data retrieved")}
                    logging.info(f"Sports data retrieved: {tool_name}")

                # 메시지 타입이 변경되었으면 더 이상 도구 메시지를 처리하지 않음
                if message_type != MessageType.TEXT:
                    return message_type, data

    # intermediate_steps도 체크 (일부 에이전트 유형은 여기에 데이터가 있을 수 있음)
    if "intermediate_steps" in result_state and not data:
        for i, step in enumerate(result_state["intermediate_steps"]):
            logging.debug(f"Step {i}: {step}")

            if len(step) < 2:
                continue

            action = step[0]
            observation = step[1]

            # 액션에서 도구 이름 추출
            tool_name = None
            tool_call_id = None

            if hasattr(action, "tool"):
                tool_name = action.tool
                # 도구 호출 ID 추출 시도
                if hasattr(action, "tool_call_id"):
                    tool_call_id = action.tool_call_id
                elif hasattr(action, "id"):
                    tool_call_id = action.id
            elif isinstance(action, dict) and "name" in action:
                tool_name = action["name"]
                # 도구 호출 ID 추출 시도
                if "tool_call_id" in action:
                    tool_call_id = action["tool_call_id"]
                elif "id" in action:
                    tool_call_id = action["id"]

            if not tool_name:
                continue

            # 디버깅: 도구 이름 로깅
            logging.debug(f"Found tool in intermediate steps: {tool_name}")

            # 도구 결과가 딕셔너리인 경우만 처리
            if not isinstance(observation, dict):
                continue

            # 도구 메시지 저장 (호출 ID가 없는 경우 생성)
            if tool_call_id is None:
                tool_call_id = f"call_{tool_name}_{datetime.now().timestamp()}"

            # 문자열로 변환 (JSON 직렬화 가능하게)
            content = str(observation.get("message", ""))
            save_tool_message(
                conversation_id=result_state.get("configurable", {}).get("thread_id", "unknown"),
                tool_call_id=tool_call_id,
                content=content,
                status="success",
                artifact=observation,
            )

            # 마켓 생성 도구
            if tool_name == "dp_asking_options":
                message_type = MessageType.MARKET_OPTIONS
                data = observation
                logging.info(f"Market options created from intermediate steps: {tool_name}")

            # 옵션 선택 도구
            elif tool_name == "set_bet_amount_dp_tool":
                message_type = MessageType.BETTING_AMOUNT_REQUEST
                data = observation
                logging.info(f"Betting amount requested from intermediate steps: {tool_name}")

            # 베팅 금액 설정 도구
            elif tool_name == "create_market_dp_tool":
                message_type = MessageType.MARKET_FINALIZED
                data = observation
                logging.info(f"Market finalized from intermediate steps: {tool_name}")

            # 스포츠 검색 도구
            elif tool_name in ["league_search", "team_search", "fixture_search"]:
                message_type = MessageType.SPORTS_SEARCH
                data = observation.get("sports_data", observation)
                logging.info(f"Sports data retrieved from intermediate steps: {tool_name}")

            # 도구 유형이 확인되면 반환
            if message_type != MessageType.TEXT:
                return message_type, data

    return message_type, data


async def process_message(user_id: str, message: str, conversation_id: str) -> dict[str, Any]:
    """
    사용자 메시지 처리

    Args:
        user_id: 사용자 ID
        message: 사용자 메시지
        conversation_id: 대화 ID

    Returns:
        처리 결과
    """
    from app.services.memory_service import get_memory_messages, save_message

    # 매번 새로운 에이전트 생성 (싱글톤 제거)
    agent = create_agent()

    # 기존 메시지 가져오기
    messages = get_memory_messages(conversation_id)

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
        final_message = result["messages"][-1] if "messages" in result and result["messages"] else None
        final_content = final_message.content if final_message and hasattr(final_message, "content") else "I couldn't process your request."

        # 메모리에 응답 저장
        save_message(conversation_id, "assistant", final_content)

        # LangChain의 ToolMessage가 있는지 확인하고 메모리에 저장
        if "messages" in result:
            from langchain_core.messages import ToolMessage
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    # ToolMessage의 정보 추출
                    tool_call_id = getattr(msg, "tool_call_id", f"call_{datetime.now().timestamp()}")
                    content = msg.content
                    name = getattr(msg, "name", "unknown_tool")
                    status = getattr(msg, "status", "success")

                    # artifact 데이터 추출
                    artifact = None
                    if isinstance(content, str):
                        try:
                            import json
                            artifact = json.loads(content)
                        except (json.JSONDecodeError, TypeError):
                            artifact = {"message": content}
                    else:
                        artifact = content

                    # 도구 메시지 저장
                    save_tool_message(
                        conversation_id=conversation_id,
                        tool_call_id=tool_call_id,
                        content=str(content),
                        status=status,
                        artifact=artifact
                    )

        # 도구 실행 결과에서 메시지 타입과 데이터 추출
        message_type, data = extract_tool_data(result)

        # 디버깅 로그 추가
        logging.info(f"Extracted message_type: {message_type}, data available: {data is not None}")

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
