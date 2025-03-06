from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

api_router = APIRouter()


@api_router.get("/", tags=["status"])
async def root():
    return {
        "status": "ok"
    }


@api_router.get("/health", tags=["status"])
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        },
        status_code=200
    )
