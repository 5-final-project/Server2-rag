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
        return json.dumps(log_record)

def setup_json_logger(logfile_path="/var/logs/server2_rag/server2_rag.log", server_name="server2-rag"):
    # always write into the host-mounted /var/logs/<server_name> folder
    logfile_path = f"/var/logs/{server_name}/{server_name}.log"
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove all existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    handler = logging.FileHandler(logfile_path)
    handler.setFormatter(JsonFormatter(server_name))
    logger.addHandler(handler)
    return logger
##