import logging

from langchain_core.tools import StructuredTool


async def token_swap_finalized(
    from_network: str,
    from_asset: str,
    amount: float,
    to_network: str,
    to_asset: str,
) -> dict:
    """
    Confirmed Token Swap information display tool.

    Args:
        from_network: Source blockchain network of the token swap
        from_asset: Source token/asset being swapped
        amount: Amount of the source asset being swapped
        to_network: Destination blockchain network for the swap
        to_asset: Destination token/asset being received

    Returns:
        Dictionary containing the swap information
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
        logging.error(f"Error token_swap_finalized: {str(e)}")
        return {"error": str(e)}


dp_token_swap_finalized = StructuredTool.from_function(
    func=token_swap_finalized,
    name="dp_token_swap_finalized",
    description="Display confirmed token swap info to the user. Be sure to obtain all necessary information from the user before using it.",
    coroutine=token_swap_finalized
)
