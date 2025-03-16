import logging
from langchain.tools import StructuredTool

from app.services.sports_service import get_fixture_details
from datetime import datetime

from app.models.market import Selection

async def get_formatted_fixture_data(fixture_id: int) -> dict:
    """
    경기 ID로 경기 정보를 조회하고 포맷팅하는 헬퍼 함수
    
    Args:
        fixture_id: 경기 ID
        
    Returns:
        포맷팅된 경기 정보 딕셔너리
    """
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
    
    return {
        "home_team_id": home_team_id,
        "home_team_name": home_team_name,
        "away_team_id": away_team_id,
        "away_team_name": away_team_name,
        "league_id": league_id,
        "league_name": league_name,
        "league_country": league_country,
        "match_date": match_date,
        "venue_name": venue_name,
        "venue_city": venue_city,
        "fixture_id": fixture_id
    }

async def asking_options(
    selections_data: list[Selection],
    fixture_id: int,
) -> dict:
    """
    마켓 옵션 선택 도구

    Args:
        selections_data: list[Selection] 선택 옵션 데이터
        fixture_id: 경기 ID

    Returns:
        FE에 표시할 Options comp data
    """
    try:
        # 공통 함수를 사용하여 경기 정보 가져오기
        fixture_info = await get_formatted_fixture_data(fixture_id)
        
        # 직렬화 가능한 딕셔너리 생성(Enum 값 직접 설정)
        serialized_data = {
            "market": {
                "title": f"{fixture_info['home_team_name']} vs {fixture_info['away_team_name']} Match Prediction",
                "description": f"Prediction market for the match between {fixture_info['home_team_name']} and {fixture_info['away_team_name']} on {fixture_info['match_date']}",
                "type": "binary",
                "status": "draft",
                "category": "sports",
                "amount": 1.0,
                "currency": "SOL",
                "close_date": fixture_info['match_date'],
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
                    "id": fixture_info['home_team_id'],
                    "name": fixture_info['home_team_name']
                },
                "away_team": {
                    "id": fixture_info['away_team_id'],
                    "name": fixture_info['away_team_name']
                },
                "league": {
                    "id": fixture_info['league_id'],
                    "name": fixture_info['league_name'],
                    "country": fixture_info['league_country']
                },
                "start_time": fixture_info['match_date'],
                "venue": {
                    "name": fixture_info['venue_name'],
                    "city": fixture_info['venue_city']
                }
            }
        }

        return serialized_data

    except Exception as e:
        logging.error(f"Error selecting option: {str(e)}")
        return {"error": str(e)}


async def asking_bet_amount(
    selection: str,
    amount: float,
    currency: str = "SOL",
) -> dict:
    """
    베팅 금액 질의 도구

    Args:
        selection: 선택한 옵션
        amount: 베팅 금액
        currency: 토큰 (기본값: SOL)

    Returns:
        FE에 표시할 베팅 금액 data
    """

    try:
        return {
            "selected_option": selection,
            "initial_amount": amount,
            "currency": currency
        }

    except Exception as e:
        logging.error(f"Error setting bet amount: {str(e)}")
        return {"error": str(e), "amount": amount, "selection": selection}

async def market_finalized(
        fixture_id: int,
        selections_data: list[Selection],
        selected_type: str,
        amount: float,
        currency: str,
) -> dict:
    """
    예측 마켓 생성 도구

    Args:
        fixture_id: 경기 ID
        selections_data: list[Selection] 선택 옵션 데이터
        selected_type: 선택한 옵션 ("win", "draw_lose")
        amount: 베팅 금액
        currency: 토큰

    Returns:
        Finalized Market Info
    """

    try:
        # 공통 함수를 사용하여 경기 정보 가져오기
        fixture_info = await get_formatted_fixture_data(fixture_id)
        
        # 직렬화 가능한 딕셔너리 생성(Enum 값 직접 설정)
        data = {
            "market": {
                "title": f"{fixture_info['home_team_name']} vs {fixture_info['away_team_name']} Match Prediction",
                "description": f"Prediction market for the match between {fixture_info['home_team_name']} and {fixture_info['away_team_name']} on {fixture_info['match_date']}",
                "type": "binary",
                "status": "draft",
                "category": "sports",
                "amount": amount,
                "currency": currency,
                "close_date": fixture_info['match_date'],
                "created_at": datetime.now().isoformat(),
            },
            "selections": [
                {"name": selection.name, "type": selection.type.value, "description": selection.description}
                for selection in selections_data
            ],
            "selected_type": selected_type,

            "event": {
                "type": "football_match",
                "fixture_id": fixture_id,
                "home_team": {"id": fixture_info['home_team_id'], "name": fixture_info['home_team_name']},
                "away_team": {"id": fixture_info['away_team_id'], "name": fixture_info['away_team_name']},
                "league": {"id": fixture_info['league_id'], "name": fixture_info['league_name'], "country": fixture_info['league_country']},
                "start_time": fixture_info['match_date'],
                "venue": {"name": fixture_info['venue_name'], "city": fixture_info['venue_city']},
            },
        }
        return data

    except Exception as e:
        logging.error(f"Error selecting option: {str(e)}")
        return {"error": str(e)}


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
