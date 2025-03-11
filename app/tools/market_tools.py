import logging
from datetime import datetime

from langchain.tools import StructuredTool

from app.models.market import SelectionType


async def create_market(
        user_id: str,
        fixture_id: int,
        home_team: str,
        away_team: str,
        league: str,
        match_date: str,
        description: str | None = None,
        bet_amount: float = 1.0
) -> dict:
    """
    예측 마켓 생성 도구

    Args:
        user_id: 사용자 ID
        fixture_id: 경기 ID
        home_team: 홈팀 이름
        away_team: 원정팀 이름
        league: 리그 이름
        match_date: 경기 날짜
        description: 마켓 설명 (옵션)
        bet_amount: 기본 베팅 금액

    Returns:
        생성된 마켓 정보
    """
    # logging.info(f"Creating market for fixture {fixture_id} - {home_team} vs {away_team}")

    try:
        # 기본 제목 생성
        title = f"{home_team} vs {away_team} Match Prediction"

        # 설명 생성
        if not description:
            description = f"Prediction market for the match between {home_team} and {away_team} on {match_date}"

        # 선택 옵션 생성 (홈팀 기준)
        selections = [
            {
                "name": f"{home_team} Win",
                "type": SelectionType.WIN,
                "description": f"{home_team} will win the match"
            },
            {
                "name": f"{home_team} Draw/Lose",
                "type": SelectionType.DRAW_LOSE,
                "description": f"{home_team} will draw or lose the match"
            }
        ]

        # 마켓 데이터 생성
        market_data = {
            "creator_id": user_id,
            "title": title,
            "description": description,
            "type": "binary",
            "category": "sports",
            "amount": bet_amount,
            "currency": "SOL",
            "close_date": match_date,
            "created_at": datetime.now().isoformat()
        }

        # 이벤트 데이터 생성
        event_data = {
            "type": "football_match",
            "fixture_id": fixture_id,
            "home_team": {
                "id": 0,  # API에서 실제 ID를 가져와야 함
                "name": home_team
            },
            "away_team": {
                "id": 0,  # API에서 실제 ID를 가져와야 함
                "name": away_team
            },
            "league": {
                "id": 0,  # API에서 실제 ID를 가져와야 함
                "name": league,
                "country": "Unknown"
            },
            "start_time": match_date,
            "venue": {
                "name": "Unknown Venue",
                "city": "Unknown City"
            }
        }

        # 마켓 패키지 생성 (USER BE에 전달할 데이터)
        market_package = {
            "market": market_data,
            "selections": selections,
            "event": event_data
        }

        return {
            "title": title,
            "description": description,
            "status": "draft",
            "selections": selections,
            "bet_amount": bet_amount,
            "fixture_id": fixture_id,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "match_date": match_date,
            "context": {
                "user_id": user_id,
                "market_package": market_package
            }
        }

    except Exception as e:
        logging.error(f"Error creating market: {str(e)}")
        return {
            "error": str(e),
            "title": f"{home_team} vs {away_team}",
            "description": "Error creating market",
            "status": "error"
        }

async def select_option(
        user_id: str,
        selection: str,
        market_title: str,
        bet_amount: float = 1.0
) -> dict:
    """
    마켓 옵션 선택 도구

    Args:
        user_id: 사용자 ID
        selection: 선택한 옵션 이름
        market_title: 마켓 제목
        bet_amount: 베팅 금액

    Returns:
        선택 결과
    """
    # logging.info(f"User {user_id} selected option: {selection} for market: {market_title}")

    try:
        return {
            "user_id": user_id,
            "selected_option": selection,
            "market_title": market_title,
            "bet_amount": bet_amount,
            "message": f"You've selected '{selection}' for '{market_title}'. The current bet amount is {bet_amount} SOL. You can change this amount if you'd like."
        }

    except Exception as e:
        logging.error(f"Error selecting option: {str(e)}")
        return {
            "error": str(e),
            "user_id": user_id,
            "selected_option": selection,
            "market_title": market_title
        }

async def set_bet_amount(
        user_id: str,
        amount: float,
        selection: str,
        market_title: str
) -> dict:
    """
    베팅 금액 설정 도구

    Args:
        user_id: 사용자 ID
        amount: 베팅 금액
        selection: 선택한 옵션
        market_title: 마켓 제목

    Returns:
        베팅 정보
    """
    # logging.info(f"User {user_id} set bet amount: {amount} SOL for {selection} on {market_title}")

    try:
        return {
            "user_id": user_id,
            "amount": amount,
            "selection": selection,
            "market_title": market_title,
            "message": f"You've set your bet amount to {amount} SOL on '{selection}' for '{market_title}'. Would you like to proceed with creating this prediction?"
        }

    except Exception as e:
        logging.error(f"Error setting bet amount: {str(e)}")
        return {
            "error": str(e),
            "user_id": user_id,
            "amount": amount,
            "selection": selection,
            "market_title": market_title
        }

# 도구 생성
create_market_tool = StructuredTool.from_function(
    func=create_market,
    name="create_market",
    description="Create a prediction market for a sports match. Use this when the user wants to create a prediction market.",
    coroutine=create_market
)

select_option_tool = StructuredTool.from_function(
    func=select_option,
    name="select_option",
    description="Process a user's selection of a prediction option. Use when the user has selected a specific prediction option.",
    coroutine=select_option
)

set_bet_amount_tool = StructuredTool.from_function(
    func=set_bet_amount,
    name="set_bet_amount",
    description="Set the bet amount for a prediction. Use when the user specifies how much they want to bet.",
    coroutine=set_bet_amount
)
