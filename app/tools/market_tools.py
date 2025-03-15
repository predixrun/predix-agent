import logging
from langchain.tools import StructuredTool

from app.services.sports_service import get_fixture_details
from datetime import datetime

from app.models.market import SelectionType, Selection

async def asking_options(
    user_id: str,
    selections_data: list[Selection],
    fixture_id: int,
) -> dict:
    """
    마켓 옵션 선택 도구

    Args:
        user_id:str 사용자 ID (필수)
        selections_data: list[Selection] 선택 옵션 데이터 (필수)
        fixture_id: 경기 ID (필수)

    Returns:
        FE에 표시할 Options comp data
    """
    try:
        # 경기 정보 조회
        fixture_data = await get_fixture_details(fixture_id)

        if not fixture_data:
            raise ValueError(f"Could not find fixture with ID: {fixture_id}")

        # 경기 정보에서 필요한 데이터 추출
        home_team_id = fixture_data["teams"]["home"]["id"]
        home_team_name = fixture_data["teams"]["home"]["name"]
        away_team_id = fixture_data["teams"]["away"]["id"]
        away_team_name = fixture_data["teams"]["away"]["name"]
        league_id = fixture_data["league"]["id"]
        league_name = fixture_data["league"]["name"]
        league_country = fixture_data["league"]["country"]
        match_date = fixture_data["fixture"]["date"]

        # 경기장 정보 가져오기
        venue_name = "Unknown Venue"
        venue_city = "Unknown City"
        if "venue" in fixture_data["fixture"] and fixture_data["fixture"]["venue"]:
            venue_name = fixture_data["fixture"]["venue"].get("name", "Unknown Venue")
            venue_city = fixture_data["fixture"]["venue"].get("city", "Unknown City")

        # 직렬화 가능한 딕셔너리 생성(Enum 값 직접 설정)
        serialized_data = {
            "market": {
                "creator_id": user_id,
                "title": f"{home_team_name} vs {away_team_name} Match Prediction",
                "description": f"Prediction market for the match between {home_team_name} and {away_team_name} on {match_date}",
                "type": "binary",
                "status": "draft",
                "category": "sports",
                "amount": 1.0,
                "currency": "SOL",
                "close_date": match_date,
                "created_at": datetime.now().isoformat()
            },
            "selections": [
                {
                    "name": selection.name,
                    "type": selection.type.value,
                    "description": selection.description
                }
                for selection in selections_data
            ],
            "event": {
                "type": "football_match",
                "fixture_id": fixture_id,
                "home_team": {
                    "id": home_team_id,
                    "name": home_team_name
                },
                "away_team": {
                    "id": away_team_id,
                    "name": away_team_name
                },
                "league": {
                    "id": league_id,
                    "name": league_name,
                    "country": league_country
                },
                "start_time": match_date,
                "venue": {
                    "name": venue_name,
                    "city": venue_city
                }
            }
        }

        return serialized_data

    except Exception as e:
        logging.error(f"Error selecting option: {str(e)}")
        return {"error": str(e), "user_id": user_id}


async def asking_bet_amount(
    selection: str,
    amount: float,
    token: str = "SOL",
) -> dict:
    """
    베팅 금액 질의 도구

    Args:
        selection: 선택한 옵션
        amount: 베팅 금액
        token: 토큰 (기본값: SOL)

    Returns:
        FE에 표시할 베팅 금액 data
    """

    try:
        return {
            "selected_option": selection,
            "initial_amount": amount,
            "token": token
        }

    except Exception as e:
        logging.error(f"Error setting bet amount: {str(e)}")
        return {"error": str(e), "amount": amount, "selection": selection}

async def market_finalized(
        user_id: str,
        fixture_id: int,
        home_team: str,
        away_team: str,
        league: str,
        match_date: str,
        bet_amount: float,
        description: str | None = None,
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
    logging.info(f"Creating market for fixture {fixture_id} - {home_team} vs {away_team}")

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

        # 응답 메시지 생성
        message = f"I've created a prediction market: '{title}'. Please select one of the options."

        return {
            "message": message,
            "message_type": "MARKET_OPTIONS",
            "data": {
                "market": market_data,
                "selections": selections,
                "event": event_data
            },
            "user_id": user_id
        }

    except Exception as e:
        logging.error(f"Error creating market: {str(e)}")
        return {
            "error": str(e),
            "message": "Error creating market",
            "message_type": "ERROR",
            "data": {
                "title": f"{home_team} vs {away_team}",
                "description": "Error creating market"
            }
        }


# 도구 생성
dp_asking_options = StructuredTool.from_function(
    func=asking_options,
    name="dp_asking_options",
    description="Generates selectable options displayed in FE based on the game content. The returned values from this tool determine the options available for the user.",
    coroutine=asking_options
)

dp_asking_bet_amount = StructuredTool.from_function(
    func=asking_bet_amount,
    name="dp_asking_bet_amount",
    description="Choose this tool when asking the user for the bet amount. It returns data for rendering the component in the FE.",
    coroutine=asking_bet_amount
)

dp_market_finalized = StructuredTool.from_function(
    func=market_finalized,
    name="dp_market_finalized",
    description="Display confirmed market informaion to the user. Be sure to obtain all necessary information from the user before using it.",
    coroutine=market_finalized
)
