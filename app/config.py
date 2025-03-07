import json
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def print_config(self):
        logger.info("[ENV] Current configuration:")
        logger.info(f"ENV: {self.ENV}")
        logger.info(f"OPENAI_API_KEY: {self.OPENAI_API_KEY[:10]}...")
        logger.info(f"ANTHROPIC_API_KEY: {self.ANTHROPIC_API_KEY[:10]}...")


settings = Settings()

# 디버깅
if settings.ENV == "DEV":
    settings.print_config()


class EmbeddingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # 1. 파라미터화된 로깅인 경우 args가 튜플 형태로 들어옴
        if isinstance(record.args, tuple):
            # args 튜플 내 entities 딕셔너리 찾기
            new_args = []
            for arg in record.args:
                if isinstance(arg, dict):
                    # embedding 필드 제거
                    arg = {k: v for k, v in arg.items() if not k.endswith('embedding')}
                new_args.append(arg)
            record.args = tuple(new_args)

        # 2. 딕셔너리 처리 유지
        elif isinstance(record.args, dict):
            filtered_args = {k: v for k, v in record.args.items() if not k.endswith('embedding')}
            record.args = filtered_args

        # 3. 문자열 처리 (이중 방어)
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            try:
                msg_dict = json.loads(record.msg)
                if isinstance(msg_dict, dict):
                    filtered_dict = {k: v for k, v in msg_dict.items() if not k.endswith('embedding')}
                    record.msg = json.dumps(filtered_dict)
            except json.JSONDecodeError:
                pass

        return True


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

    # 임베딩 필터
    embedding_filter = EmbeddingFilter()
    file_handler.addFilter(embedding_filter)
    console_handler.addFilter(embedding_filter)

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
