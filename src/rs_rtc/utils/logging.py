import logging
import os
from datetime import datetime

from rs_rtc import RESULTS_DIR

LOG_DIR = RESULTS_DIR / "logs"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logger(logger_name, run_name):
    """콘솔과 results/logs/{run_name}_{시각}_{pid}.log 에 동시에 기록하는 로거를 만든다."""
    # 로그 파일 경로 준비
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = LOG_DIR / f"{run_name}_{timestamp}_{os.getpid()}.log"

    # 로거 기본 설정
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(LOG_FORMAT)

    # 콘솔 출력 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 파일 출력 핸들러
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("로그 파일: %s", log_file_path)
    return logger
