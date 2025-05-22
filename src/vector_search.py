"""
벡터 검색 모듈 - 랭그래프와 호환되는 벡터 검색 기능 제공
"""
import httpx
import time
from typing import List, Dict, Any, Tuple
from src.config import get_settings

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
    
    start = time.time()
    try:
        with httpx.Client(timeout=settings.TIMEOUT_SEC) as client:
            resp = client.post(_API, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", []), time.time() - start
    except Exception as e:
        # 오류 발생 시 로깅 및 빈 결과 반환
        print(f"Vector search failed: {str(e)}")
        return [], time.time() - start
