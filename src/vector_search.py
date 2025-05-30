"""
벡터 검색 모듈 - 랭그래프와 호환되는 벡터 검색 기능 제공
"""
import httpx
import time
import logging
import uuid
from typing import List, Dict, Any, Tuple
from src.config import get_settings
from .logging_utils import setup_json_logger

# Prometheus 메트릭 임포트
from prometheus_client import Counter, Histogram

setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")

settings = get_settings()
_API = settings.VECTOR_API_URL.rstrip("/") + "/search"

from .api import (
    team5_vector_searches,
    team5_vector_search_duration,
    team5_vector_search_errors,
    team5_vector_search_results
)
def search_vector(keywords: List[str], k: int = None) -> Tuple[List[Dict[str, Any]], float]:
    """
    벡터 검색 API를 호출하여 관련 문서를 검색합니다.
    
    Args:
        keywords: 검색할 키워드 리스트
        k: 반환할 최대 문서 수 (기본값: 설정값 사용)
        
    Returns:
        Tuple[검색 결과 문서 리스트, 처리 시간(초)]
    """
    k = k or settings.VECTOR_TOP_K
    payload = {
        "keywords": keywords, 
        "k": k, 
        "filter": {}
    }
    trace_id = str(uuid.uuid4())
    start = time.time()
    service_name = "server2-rag"
    
    try:
        logger.info({
            "event": "vector_search_start",
            "trace_id": trace_id,
            "keywords": keywords,
            "k": k
        })
        
        # 벡터 검색 요청 메트릭 증가
        team5_vector_searches.labels(service=service_name).inc()
        
        with httpx.Client(timeout=settings.TIMEOUT_SEC) as client:
            resp = client.post(_API, json=payload)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - start
            results = data.get("results", [])
            
            # 메트릭 기록
            team5_vector_search_duration.labels(service=service_name).observe(elapsed)
            team5_vector_search_results.labels(service=service_name).observe(len(results))
            
            logger.info({
                "event": "vector_search_success",
                "trace_id": trace_id,
                "elapsed_time": elapsed,
                "num_results": len(results)
            })
            return results, elapsed
            
    except httpx.HTTPStatusError as e:
        elapsed = time.time() - start
        error_type = f"http_{e.response.status_code}"
        team5_vector_search_errors.labels(service=service_name, error_type=error_type).inc()
        team5_vector_search_duration.labels(service=service_name).observe(elapsed)
        
        logger.error({
            "event": "vector_search_http_error",
            "trace_id": trace_id,
            "error": str(e),
            "status_code": e.response.status_code,
            "elapsed_time": elapsed
        })
        return [], elapsed
        
    except httpx.TimeoutException as e:
        elapsed = time.time() - start
        team5_vector_search_errors.labels(service=service_name, error_type="timeout").inc()
        team5_vector_search_duration.labels(service=service_name).observe(elapsed)
        
        logger.error({
            "event": "vector_search_timeout",
            "trace_id": trace_id,
            "error": str(e),
            "elapsed_time": elapsed
        })
        return [], elapsed
        
    except Exception as e:
        elapsed = time.time() - start
        error_type = type(e).__name__
        team5_vector_search_errors.labels(service=service_name, error_type=error_type).inc()
        team5_vector_search_duration.labels(service=service_name).observe(elapsed)
        
        logger.error({
            "event": "vector_search_error",
            "trace_id": trace_id,
            "error": str(e),
            "elapsed_time": elapsed
        })
        return [], elapsed