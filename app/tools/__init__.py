# 도구 모듈 초기화
from app.config import logger


def initialize_agent():
    """
    에이전트 초기화 - 사전 검증용으로만 사용
    (실제 에이전트는 매 요청마다 새로 생성)
    """
    logger.info("Testing ReAct agent creation")

    # 테스트용으로 한 번 에이전트 생성
    from app.agent import create_agent
    try:
        agent = create_agent()
        logger.info("ReAct agent test creation successful")
    except Exception as e:
        logger.error(f"Error testing agent creation: {e}", exc_info=True)
        raise

# 도구들을 내보냄
from .market_tools import create_market_dp_tool, dp_asking_options, set_bet_amount_dp_tool
from .sports_tools import fixture_search_tool, league_search_tool, team_search_tool

__all__ = [
    'initialize_agent',
    'league_search_tool',
    'team_search_tool',
    'fixture_search_tool',
    'create_market_dp_tool',
    'dp_asking_options',
    'set_bet_amount_dp_tool',
]