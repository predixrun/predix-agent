from fastapi import FastAPI, HTTPException, Request

from app.config import logger, setup_logging
from app.models.response_models import TemplateJSONResponse
from app.routers.api import api_router
from app.tools import initialize_agent

setup_logging()


def create_app() -> FastAPI:
    app = FastAPI(
        title="PrediX Agent Server",
        version="0.1.0",
        default_response_class=TemplateJSONResponse
    )
    initialize_agent()
    logger.info("LangChain tools initialized")

    app.include_router(api_router)
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
