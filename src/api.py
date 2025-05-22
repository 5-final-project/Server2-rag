"""
FastAPI 엔드포인트: POST /process
- 입력: { "text": "회의 전체 텍스트" }
- 출력: { "result": [ ... chunk별 RAG 결과 ... ] }
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import logging
import sys
import time
import uuid
from .graph import run_pipeline
from .logging_utils import setup_json_logger
from fastapi import Request

# JSON Logger 설정 (logs/server2_rag.log)
setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")

app = FastAPI()

class ProcessRequest(BaseModel):
    text: str

@app.post("/process")
async def process(req: ProcessRequest, request: Request) -> Dict[str, Any]:
    trace_id = str(uuid.uuid4())
    start = time.time()
    logger.info({
        "event": "process_request_received",
        "trace_id": trace_id,
        "client_host": request.client.host if request.client else None,
        "text_length": len(req.text)
    })
    if not req.text.strip():
        logger.warning({
            "event": "empty_text_request",
            "trace_id": trace_id
        })
        raise HTTPException(status_code=400, detail="text 필드는 비어 있을 수 없습니다.")
    try:
        logger.info({
            "event": "run_pipeline_start",
            "trace_id": trace_id
        })
        pipeline_result = run_pipeline(req.text)
        elapsed = time.time() - start
        logger.info({
            "event": "run_pipeline_completed",
            "trace_id": trace_id,
            "elapsed_time": elapsed,
            "result_keys": list(pipeline_result.keys())
        })
        return pipeline_result
    except Exception as e:
        elapsed = time.time() - start
        logger.error({
            "event": "run_pipeline_error",
            "trace_id": trace_id,
            "error": str(e),
            "elapsed_time": elapsed
        })
        raise
