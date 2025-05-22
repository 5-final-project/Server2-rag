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

setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")

settings = get_settings()
_API = settings.VECTOR_API_URL.rstrip("/") + "/search"

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
    try:
        logger.info({
            "event": "vector_search_start",
            "trace_id": trace_id,
            "keywords": keywords,
            "k": k
        })
        with httpx.Client(timeout=settings.TIMEOUT_SEC) as client:
            resp = client.post(_API, json=payload)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - start
            results = data.get("results", [])
            logger.info({
                "event": "vector_search_success",
                "trace_id": trace_id,
                "elapsed_time": elapsed,
                "num_results": len(results)
            })
            return results, elapsed
    except Exception as e:
        elapsed = time.time() - start
        logger.error({
            "event": "vector_search_error",
            "trace_id": trace_id,
            "error": str(e),
            "elapsed_time": elapsed
        })
        return [], elapsed
