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

# 하이브리드 재랭킹 검색 엔드포인트 사용 (가장 고급 검색)
_API = settings.VECTOR_API_URL.rstrip("/") + "/search/hybrid-reranked"

from .metrics import (
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
    
    # 키워드 리스트를 문자열로 변환 (벡터 검색 서버가 기대하는 형식)
    query_text = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
    
    # 벡터 검색 서버의 실제 API 형식에 맞게 페이로드 구성
    # 브라우저 스크린샷에서 확인한 API 문서 형식 사용
    payload = {
        "query": query_text,
        "top_k": k,
        "indices": []  # 빈 배열이면 마스터 인덱스에서 검색
    }
    
    trace_id = str(uuid.uuid4())
    start = time.time()
    service_name = "server2-rag"
    
    try:
        logger.info({
            "event": "vector_search_start",
            "trace_id": trace_id,
            "keywords": keywords,
            "query_text": query_text,
            "k": k,
            "api_url": _API
        })
        
        # 벡터 검색 요청 메트릭 증가
        team5_vector_searches.labels(service=service_name).inc()
        
        # POST 요청으로 검색 수행
        with httpx.Client(timeout=settings.TIMEOUT_SEC) as client:
            resp = client.post(_API, json=payload, headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            })
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - start
            
            # 응답 형식에 따라 결과 추출
            # API 문서에서 확인한 형식: {"results": [...]}
            if "results" in data:
                results = data["results"]
            elif isinstance(data, list):
                results = data
            else:
                logger.warning(f"Unexpected response format: {data}")
                results = []
            
            # 결과를 우리 형식에 맞게 변환
            formatted_results = []
            for item in results:
                if isinstance(item, dict):
                    formatted_item = {
                        "page_content": item.get("page_content", ""),
                        "metadata": item.get("metadata", {}),
                        "score": item.get("score", 0.0)
                    }
                    formatted_results.append(formatted_item)
            
            # 메트릭 기록
            team5_vector_search_duration.labels(service=service_name).observe(elapsed)
            team5_vector_search_results.labels(service=service_name).observe(len(formatted_results))
            
            logger.info({
                "event": "vector_search_success",
                "trace_id": trace_id,
                "elapsed_time": elapsed,
                "num_results": len(formatted_results),
                "response_keys": list(data.keys()) if isinstance(data, dict) else "list_response",
                "first_result_preview": formatted_results[0] if formatted_results else None
            })
            return formatted_results, elapsed
            
    except httpx.HTTPStatusError as e:
        elapsed = time.time() - start
        error_type = f"http_{e.response.status_code}"
        team5_vector_search_errors.labels(service=service_name, error_type=error_type).inc()
        team5_vector_search_duration.labels(service=service_name).observe(elapsed)
        
        try:
            error_detail = e.response.text
        except:
            error_detail = str(e)
        
        logger.error({
            "event": "vector_search_http_error",
            "trace_id": trace_id,
            "error": str(e),
            "error_detail": error_detail,
            "status_code": e.response.status_code,
            "elapsed_time": elapsed,
            "api_url": _API,
            "payload": payload
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
            "elapsed_time": elapsed,
            "api_url": _API,
            "timeout_seconds": settings.TIMEOUT_SEC
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
            "elapsed_time": elapsed,
            "api_url": _API,
            "payload": payload
        })
        return [], elapsed