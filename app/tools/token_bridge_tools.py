import logging

from langchain_core.tools import StructuredTool


async def token_bridge_finalized(
    from_network: str,
    from_asset: str,
    amount: float,
    to_network: str,
    to_asset: str,
) -> dict:
    """
    Confirmed Token Bridge information display tool.

    Args:
        from_network: Source blockchain network of the token bridge
        from_asset: Source token/asset being bridged
        amount: Amount of the source asset being bridged
        to_network: Destination blockchain network for the bridge
        to_asset: Destination token/asset being received

    Returns:
        Dictionary containing the bridge information
    """

    try:
        data = {
            "from_network": from_network,
            "from_asset": from_asset,
            "amount": amount,
            "to_network": to_network,
            "to_asset": to_asset,
        }
        return data

    except Exception as e:
        logging.error(f"Error token_bridge_finalized: {str(e)}")
        return {"error": str(e)}


dp_token_bridge_finalized = StructuredTool.from_function(
    func=token_bridge_finalized,
    name="dp_token_bridge_finalized",
    description="Display confirmed token bridge info to the user. Be sure to obtain all necessary information from the user before using it.",
    coroutine=token_bridge_finalized
)
