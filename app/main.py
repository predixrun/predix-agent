import logging
import os

from fastapi import FastAPI, HTTPException, Request

from app.config import setup_logging
from app.models.response_models import TemplateJSONResponse
from app.routers.api import api_router

setup_logging()

# 데이터 디렉토리
os.makedirs(os.path.join(os.getcwd(), "data", "memory"), exist_ok=True)

def create_app() -> FastAPI:
    app = FastAPI(
        title="PrediX Agent Server",
        version="0.3.0",
        default_response_class=TemplateJSONResponse,
    )
    app.include_router(api_router)
    logging.info("LangGraph Predix agent initialized successfully")
    return app

app = create_app()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return TemplateJSONResponse(
        status_code=exc.status_code,
        content={
            "status": "FAIL",
            "errorCode": exc.status_code,
            "data": exc.detail,
        }
    )
