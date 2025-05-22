import sys, json, time
import logging
import uuid
from src.config import get_settings
from src.graph import run_pipeline
from .logging_utils import setup_json_logger

setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")

def main() -> None:
    text = sys.stdin.read()          # 단일 문자열 STDIN 입력
    trace_id = str(uuid.uuid4())
    start = time.time()
    logger.info({
        "event": "main_pipeline_start",
        "trace_id": trace_id,
        "input_length": len(text)
    })
    try:
        result = run_pipeline(text)
        elapsed = time.time() - start
        result["_elapsed_total"] = elapsed
        logger.info({
            "event": "main_pipeline_completed",
            "trace_id": trace_id,
            "elapsed_time": elapsed,
            "result_keys": list(result.keys())
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        elapsed = time.time() - start
        logger.error({
            "event": "main_pipeline_error",
            "trace_id": trace_id,
            "error": str(e),
            "elapsed_time": elapsed
        })
        raise

if __name__ == "__main__":
    main()
