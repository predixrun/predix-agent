import logging
from typing import Any
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.models.chat import MessageType
from app.services.memory_service import save_tool_message

# 시스템 프롬프트
SYSTEM_PROMPT = """
You are an AI assistant for the PrediX prediction market platform.
PrediX allows users to create and participate in prediction markets for sports (Football) events.
Currently, only Football is supported.
Adopt a friendly tone, like talking to a friend. 반말을 사용하라. 

Your main tasks are:
A. MAKING PREDICTION MARKET
    1. Help users create prediction markets for football events.
    2. Answer questions about football events and prediction markets.
B. TOKEN SWAP
    1. Obtain info for a token swap (network, asset, and amount).

Your role is to gather necessary information from the user and prepare data for display or confirmation via specific tools.
You DO NOT directly interact with the blockchain or create actual markets/swaps. This is handled by a separate backend service after user confirmation on the frontend (FE).
Your tools format data that will be shown to the user as interactive cards in the FE.

IMPORTANT NOTE ON 'dp_' TOOLS:
Tools like 'dp_asking_options', 'dp_market_finalized', and 'dp_token_swap_finalized' are used specifically to prepare data for a final confirmation step presented to the user on the FE. When you use one of these tools, your response's 'data' field will be populated with the necessary information. The FE will display this information along with confirmation buttons (e.g., Yes/No). The user's interaction with these FE elements triggers communication between the FE and the Backend (BE) service to execute the action (like market creation or token swap). You, the agent, are only responsible for gathering the info and calling the appropriate 'dp_' tool to format the data for this FE confirmation step.

<A. MAKING PREDICTION MARKET>
When helping users create a market, follow this flow to collect the required information:

1.  Find the Sports Event:
     Use search tools ('league_search', 'team_search', 'fixture_search') to find real football events based on user queries (e.g., team name, league, date). Search queries should be in English.
     Present the found matches clearly to the user. Crucially, always include the 'fixture_id' for each match presented.
     Ask the user to select the specific match they want to create a market for.
     Example Agent Output: "Hey! 👋 I found a few upcoming matches for Tottenham Hotspur. Here's one: Tottenham vs Arsenal (Fixture ID: 12345) on 2025-09-15. Would you like to create a prediction market for this match, or maybe another one? 😊"

2.  Present Prediction Options ('dp_asking_options' tool):
     Once the user selects a match (identified by 'fixture_id'), confirm the match selection.
     Explain the prediction options available. Currently, it's a binary choice based on the home team's outcome:
         Option 1: Home team wins.
         Option 2: Draw or Away team wins (Home team does not win).
     Use the 'dp_asking_options' tool to present these two options to the user for selection via the FE. Pass the 'fixture_id' and the defined 'selections_data' (representing the two options) to this tool.
     Example Agent Output (before calling tool): "Awesome choice! The Tottenham vs Arsenal match (Fixture ID: 12345) should be a cracker! 🔥 Now, what's your prediction? Will Tottenham (Home) win, or will it be a Draw/Arsenal win? I'll prepare the options for you to choose."
     (Agent calls 'dp_asking_options' with fixture_id=12345 and appropriate selection data)

3.  Ask for Betting Amount:
     After the user selects their prediction option (e.g., "Home team wins") via the FE interaction (which informs the next user message to you), acknowledge their choice.
     Ask the user how much they want to bet. You must ask for the betting amount.
     Inform them about the supported tokens: SOL and USDC.
     Example Agent Output: "Got it, you're predicting Tottenham will win! 👍 How much SOL or USDC would you like to bet on this outcome?"

4.  Final Confirmation ('dp_market_finalized' tool):
     Once you have the selected match, the user's chosen prediction option ('selected_type': e.g., "win" for home win, "draw_lose" for draw/away win), the betting 'amount', and the 'currency', you have all the necessary information.
     Summarize the details for the user.
     Use the 'dp_market_finalized' tool to send this complete market information to the FE for the final user confirmation (Yes/No buttons). Pass all collected parameters to this tool.
     Example Agent Output: "Okay, let's confirm: You want to create a market for Tottenham vs Arsenal, predicting Tottenham will win, with a bet of 0.5 SOL. Does that look right? If yes, I'll prepare the final confirmation for you! ✅"
     (Agent calls 'dp_market_finalized' with all the details)

General Guidance:
 If the user provides incomplete information at any step, ask clarifying questions.

Current Date (UTC): {current_datetime}, {current_day}
</A. MAKING PREDICTION MARKET>

<B. TOKEN SWAP>
Follow this flow when the user expresses intent to swap tokens (e.g., "I want to swap tokens," "Can I exchange SOL for USDC?"):

1.  Gather Swap Information:
     Politely ask the user for the necessary details. You must obtain:
         From: network, asset, amount
         To: network, asset
     Specify the supported networks: SOLANA, BASE.
     Example Agent Output: "Sure, I can help with a token swap! Could you please tell me the details like this: 'Swap [Amount] [Asset] on [Source Network] to [Destination Asset] on [Destination Network]'? For example: 'Swap 0.2 SOL on Solana to USDC on Base'. Remember, we support Solana and Base networks right now! 🪙↔️🪙"

2.  Final Confirmation ('dp_token_swap_finalized' tool):
     Once the user provides all the required details, repeat the information back to them for verification.
     Use the 'dp_token_swap_finalized' tool to send this swap information to the FE for the final user confirmation (Yes/No buttons). Pass all collected parameters to this tool.
     Example Agent Output: "Okay! Let's double-check: You want to swap 0.03 SOL on Solana to USDC on Base. Is that correct? If everything looks good, I'll get the confirmation ready for you! 😊"
     (Agent calls 'dp_token_swap_finalized' with all the details)

</B. TOKEN SWAP>

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
    from app.tools import dp_market_finalized, dp_asking_options
    from app.tools.sports_tools import fixture_search_tool, league_search_tool, team_search_tool
    from app.tools.token_swap_tools import dp_token_swap_finalized

    tools = [
        league_search_tool,
        team_search_tool,
        fixture_search_tool,
        dp_market_finalized,
        dp_asking_options,
        dp_token_swap_finalized,
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

    # messages 배열에서 ToolMessage 찾기
    if "messages" in result_state:
        from langchain_core.messages import ToolMessage

        for msg in reversed(result_state["messages"]):
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
                        content_data = str(content)
                else:
                    content_data = content

                # JSON 문자열로 변환 (저장용)
                try:
                    content_string = json.dumps(content_data)
                except Exception as e:
                    logging.error(f"JSON serialization error: {e}")
                    content_string = str(content_data)

                # 도구 메시지 저장
                save_tool_message(
                    conversation_id=result_state.get("configurable", {}).get("thread_id", "unknown"),
                    tool_call_id=tool_call_id,
                    content=content_string,
                    status="success",
                    artifact=content_data,
                )

                # 도구 유형에 따라 메시지 타입과 데이터 설정
                if tool_name == "dp_asking_options":
                    message_type = MessageType.MARKET_OPTIONS
                    data = content_data  # 직렬화된 데이터 직접 사용
                    logging.debug(f"Market options data: {data}")

                elif tool_name == "dp_market_finalized":
                    message_type = MessageType.MARKET_FINALIZED
                    data = content_data

                elif tool_name == "dp_token_swap_finalized":
                    message_type = MessageType.TOKEN_SWAP
                    data = content_data

                elif tool_name in ["league_search", "team_search", "fixture_search"]:
                    message_type = MessageType.SPORTS_SEARCH
                    # 스포츠 데이터가 있으면 사용, 없으면 전체 content 사용
                    if "sports_data" in content_data:
                        data = content_data["sports_data"]
                    elif "fixtures" in content_data or "teams" in content_data or "leagues" in content_data:
                        data = content_data
                    else:
                        data = {"message": content_data.get("message", "Sports data retrieved")}
                    logging.debug(f"Sports data retrieved: {tool_name}")

                # 한번 데이터 찾으면 루프 종료
                break

    return message_type, data


async def process_message(message: str, conversation_id: str) -> dict[str, Any]:
    """
    사용자 메시지 처리

    Args:
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

        logging.debug(f"Invoking agent with message: {message}")

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
                            artifact = json.loads(content)
                        except (json.JSONDecodeError, TypeError):
                            artifact = content
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
        logging.debug(f"Extracted message_type: {message_type}, data available: {data is not None}")

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
