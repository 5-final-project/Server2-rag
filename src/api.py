"""
FastAPI 엔드포인트: POST /process
- 입력: { "text": "회의 전체 텍스트" }
- 출력: { "result": [ ... chunk별 RAG 결과 ... ] }
"""
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, List
import logging
import sys
import time
import uuid
from .graph import run_pipeline
from .logging_utils import setup_json_logger

# Prometheus 관련 임포트
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import Response

# 모든 메트릭을 metrics 모듈에서 임포트 (순환 임포트 방지)
from .metrics import (
    team5_rag_requests,
    team5_rag_duration, 
    team5_rag_errors,
    team5_rag_chunk_count,
    team5_rag_active_requests,
    team5_gpu_utilization,
    team5_gpu_memory_used,
    team5_gpu_memory_total,
    team5_gpu_temperature,
    team5_gpu_power_usage,
    get_gpu_metrics_with_retry
)

# JSON Logger 설정 (/var/logs/server2_rag/server2_rag.log)
setup_json_logger("/var/logs/server2_rag/server2_rag.log", "server2-rag")
logger = logging.getLogger()

app = FastAPI(title="Server2 RAG Pipeline", version="1.0.0")

class ProcessRequest(BaseModel):
    text: str

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """메트릭 수집을 위한 미들웨어 (GPU 메트릭 포함)"""
    start_time = time.time()
    endpoint = request.url.path
    service_name = "server2-rag"
    
    # /metrics 엔드포인트는 메트릭에서 제외
    if endpoint == "/metrics":
        return await call_next(request)
    
    # 활성 요청 수 증가
    team5_rag_active_requests.labels(service=service_name).inc()
    
    try:
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        # GPU 모니터링은 응답 처리 후에 안전하게 수행
        if endpoint == "/process":  # 주요 엔드포인트에서만 GPU 메트릭 수집
            try:
                gpu_metrics = get_gpu_metrics_with_retry(max_retries=3)
                if gpu_metrics:
                    team5_gpu_utilization.labels(service=service_name).set(gpu_metrics['utilization'])
                    team5_gpu_memory_used.labels(service=service_name).set(gpu_metrics['memory_used'])
                    team5_gpu_memory_total.labels(service=service_name).set(gpu_metrics['memory_total'])
                    team5_gpu_temperature.labels(service=service_name).set(gpu_metrics['temperature'])
                    team5_gpu_power_usage.labels(service=service_name).set(gpu_metrics['power_usage'])
            except Exception as e:
                # GPU 모니터링 실패해도 로깅만 남기고 계속
                logger.debug(f"GPU monitoring failed after retries: {e}")
        
        # 요청 수 증가
        team5_rag_requests.labels(service=service_name, endpoint=endpoint).inc()
        
        # 처리 시간 기록
        team5_rag_duration.labels(service=service_name, endpoint=endpoint).observe(processing_time)
        
        # 에러 응답인 경우 에러 카운터 증가
        if response.status_code >= 400:
            error_type = f"http_{response.status_code}"
            team5_rag_errors.labels(service=service_name, endpoint=endpoint, error_type=error_type).inc()
        
        return response
    except Exception as e:
        processing_time = time.time() - start_time
        
        # 예외 발생 시 에러 카운터 증가
        error_type = type(e).__name__
        team5_rag_errors.labels(service=service_name, endpoint=endpoint, error_type=error_type).inc()
        
        # 처리 시간은 여전히 기록
        team5_rag_duration.labels(service=service_name, endpoint=endpoint).observe(processing_time)
        
        raise
    finally:
        # 활성 요청 수 감소
        team5_rag_active_requests.labels(service=service_name).dec()

@app.post("/process")
async def process(req: ProcessRequest, request: Request) -> Dict[str, Any]:
    trace_id = str(uuid.uuid4())
    start = time.time()
    service_name = "server2-rag"
    
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
        # 에러 메트릭 증가
        team5_rag_errors.labels(service=service_name, endpoint="/process", error_type="empty_text").inc()
        raise HTTPException(status_code=400, detail="text 필드는 비어 있을 수 없습니다.")
    
    try:
        logger.info({
            "event": "run_pipeline_start",
            "trace_id": trace_id
        })
        
        # RAG 파이프라인 실행
        pipeline_result = run_pipeline(req.text)
        
        elapsed = time.time() - start
        
        # 처리된 청크 수 메트릭 기록
        if isinstance(pipeline_result, dict) and "result" in pipeline_result:
            chunk_count = len(pipeline_result["result"])
            team5_rag_chunk_count.labels(service=service_name).observe(chunk_count)
        
        logger.info({
            "event": "run_pipeline_completed",
            "trace_id": trace_id,
            "elapsed_time": elapsed,
            "result_keys": list(pipeline_result.keys()) if isinstance(pipeline_result, dict) else []
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
        
        # 에러 메트릭 증가
        error_type = type(e).__name__
        team5_rag_errors.labels(service=service_name, endpoint="/process", error_type=error_type).inc()
        
        raise

@app.get("/metrics")
async def get_metrics():
    """Prometheus 메트릭 엔드포인트"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """헬스체크 엔드포인트"""
    return {
        "message": "Server2 RAG Pipeline is running",
        "service": "server2-rag",
        "status": "healthy"
    }

# Prometheus FastAPI Instrumentator 설정 (기본 메트릭)
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="team5_rag_inprogress",
    inprogress_labels=True,
)

# FastAPI 앱에 instrumentator 적용
instrumentator.instrument(app).expose(app)