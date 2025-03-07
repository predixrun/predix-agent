from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import settings

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME)


def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != f"ApiKey {settings.API_KEY}":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
