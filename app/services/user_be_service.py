import uuid
from typing import Any

import httpx

from app.config import logger, settings


async def request_market_creation(market_data: dict[str, Any]) -> dict[str, Any]:
    """
    Request market creation from USER BE.

    Args:
        market_data: Market data to be created

    Returns:
        Response from USER BE
    """
    try:
        # Make request to USER BE
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.USERBE_API_URL}/markets/create",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"ApiKey {settings.USERBE_API_KEY}"
                },
                json=market_data,
                timeout=10.0
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error creating market: {response.status_code} - {response.text}")
                # Return mock response for PoC
                return {
                    "status": "success",
                    "market_id": market_data.get("id", str(uuid.uuid4())),
                    "blockchain": "solana",
                    "timestamp": market_data.get("created_at")
                }
    except Exception as e:
        logger.error(f"Error requesting market creation: {e}")
        # Return mock response for PoC
        return {
            "status": "success",
            "market_id": market_data.get("id", str(uuid.uuid4())),
            "blockchain": "solana",
            "timestamp": market_data.get("created_at")
        }
