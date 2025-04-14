import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

import pytz
from pydantic_settings import BaseSettings

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    ENV: str = "DEV"

    API_KEY: str

    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str

    SPORTS_API_KEY: str

    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY : str
    LANGSMITH_PROJECT: str = "predix"

    GAME_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def setup_logging():
    kst = pytz.timezone('Asia/Seoul')

    class KSTFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, kst)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.isoformat()

    formatter = KSTFormatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )

    log_file = 'logs/app.log'
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 루트 로거
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # FastAPI 로거
    for logger_name in ['uvicorn', 'uvicorn.access']:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
