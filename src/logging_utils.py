import os
import logging
from datetime import datetime
from pythonjsonlogger import jsonlogger
import json

def ensure_log_dir(log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

class JsonFormatter(logging.Formatter):
    def __init__(self, server_name):
        super().__init__()
        self.server_name = server_name
    def format(self, record):
        log_record = {
            "@timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "server": self.server_name,
        }
        msg = record.getMessage()
        msg_dict = None
        try:
            if isinstance(record.msg, dict):
                msg_dict = record.msg
            elif isinstance(msg, str) and msg.startswith("{"):
                msg_dict = json.loads(msg)
        except Exception:
            msg_dict = None
        if isinstance(msg_dict, dict):
            log_record.update(msg_dict)
        else:
            log_record["msg"] = msg
        # ensure_ascii=False를 추가하여 한글이 유니코드 이스케이프 시퀀스로 변환되지 않도록 함
        return json.dumps(log_record, ensure_ascii=False)

def setup_json_logger(logfile_path="logs/server2_rag.log", server_name="server2-rag"):
    sanitized_name = server_name.replace("-", "_")
    
    # 로컬 로그 디렉토리 생성 및 설정
    local_log_dir = "logs"
    if not os.path.exists(local_log_dir):
        os.makedirs(local_log_dir)
    local_logfile_path = f"{local_log_dir}/{sanitized_name}.log"
    
    # 컨테이너 내부 로그 경로 (기존)
    container_logfile_path = f"/var/logs/{sanitized_name}/{sanitized_name}.log"
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    
    # 로컬 로그 파일 핸들러 추가
    try:
        local_handler = logging.FileHandler(local_logfile_path, encoding='utf-8')
        local_handler.setFormatter(JsonFormatter(server_name))
        logger.addHandler(local_handler)
        print(f"Local log file setup at: {local_logfile_path}")
    except Exception as e:
        print(f"Warning: Could not set up local log file: {e}")
    
    # 컨테이너 로그 파일 핸들러 추가 (기존)
    try:
        os.makedirs(os.path.dirname(container_logfile_path), exist_ok=True)
        container_handler = logging.FileHandler(container_logfile_path, encoding='utf-8')
        container_handler.setFormatter(JsonFormatter(server_name))
        logger.addHandler(container_handler)
        print(f"Container log file setup at: {container_logfile_path}")
    except Exception as e:
        print(f"Warning: Could not set up container log file: {e}")
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonFormatter(server_name))
    logger.addHandler(console_handler)
    
    return logger
##