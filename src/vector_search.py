"""
벡터 검색 모듈 - 랭그래프와 호환되는 벡터 검색 기능 제공
"""
import httpx
import time
import logging
import uuid
import os
from typing import List, Dict, Any, Tuple
from src.config import get_settings
from .logging_utils import setup_json_logger

# Prometheus 메트릭 임포트
from prometheus_client import Counter, Histogram

setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")

settings = get_settings()
# 설정에서 벡터 검색 API URL 가져오기
_API = settings.VECTOR_API_URL
if not _API.endswith("/search/hybrid-reranked"):
    _API = _API + "/search/hybrid-reranked"
logger.info(f"벡터 검색 API URL 설정: {_API}")

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
    # 새로운 요청 형식에 맞게 변경
    payload = {
        "query": " ".join(keywords),
        "top_k": k,
        "indices": ["master_documents"]  # 기본 인덱스 설정 추가
    }
    
    trace_id = str(uuid.uuid4())
    start = time.time()
    service_name = "server2-rag"
    
    logger.info({
        "event": "vector_search_start",
        "trace_id": trace_id,
        "keywords": keywords,
        "k": k,
        "vector_api_url": _API,
        "payload": payload
    })
    
    # 키워드가 비어있으면 빈 결과 반환
    if not keywords:
        logger.warning({
            "event": "vector_search_empty_keywords",
            "trace_id": trace_id,
            "message": "검색 키워드가 비어 있습니다."
        })
        team5_vector_search_errors.labels(service=service_name, error_type="empty_keywords").inc()
        return [], time.time() - start
    
    try:
        # 벡터 검색 요청 메트릭 증가
        team5_vector_searches.labels(service=service_name).inc()
        
        # 샘플 테스트 문서 반환 (API 서버가 없을 경우를 대비)
        if os.path.exists("data/sample_doc.txt") and False:  # 실제 API 호출로 변경하기 위해 False 추가
            logger.info({
                "event": "vector_search_using_sample",
                "trace_id": trace_id,
                "message": "벡터 API 서버가 실행되지 않아 샘플 데이터 사용"
            })
            elapsed = time.time() - start
            
            # 샘플 문서 읽기
            with open("data/sample_doc.txt", "r", encoding="utf-8") as f:
                content = f.read()
                
            # 테스트용 결과 생성
            results = [{
                "page_content": content,
                "metadata": {
                    "source": "sample_doc.txt", 
                    "title": "IT 인프라 장애 사례", 
                    "date": "2023-03-10"
                },
                "score": 0.95
            }]
            
            # 메트릭 기록
            team5_vector_search_duration.labels(service=service_name).observe(elapsed)
            team5_vector_search_results.labels(service=service_name).observe(len(results))
            
            logger.info({
                "event": "vector_search_sample_success",
                "trace_id": trace_id,
                "elapsed_time": elapsed,
                "num_results": len(results)
            })
            return results, elapsed
        
        with httpx.Client(timeout=settings.TIMEOUT_SEC) as client:
            logger.debug({
                "event": "vector_search_request",
                "trace_id": trace_id,
                "url": _API,
                "payload": payload
            })
            
            resp = client.post(_API, json=payload)
            resp.raise_for_status()
            raw_data = resp.text
            logger.info({
                "event": "vector_search_raw_response",
                "trace_id": trace_id,
                "raw_response": raw_data[:500] if len(raw_data) > 500 else raw_data  # 응답이 너무 길면 일부만 로깅
            })
            
            data = resp.json()
            elapsed = time.time() - start
            
            # 응답 형식에 맞게 결과 변환
            raw_results = data.get("results", [])
            logger.info({
                "event": "vector_search_parsed_results",
                "trace_id": trace_id,
                "raw_results_count": len(raw_results),
                "raw_results_sample": str(raw_results[:1])[:200] if raw_results else "[]"
            })
            
            results = []
            
            # 결과 형식 변환
            for doc in raw_results:
                # 새로운 문서 객체 생성
                processed_doc = {}
                
                # API 명세에 맞는 응답 구조 처리
                # page_content는 이미 응답에 포함되어 있으므로 바로 추출
                if "page_content" in doc:
                    processed_doc["page_content"] = doc["page_content"]
                # 이전 구조 호환성 유지
                elif "chunk_en" in doc:
                    processed_doc["page_content"] = doc["chunk_en"]
                elif "chunk" in doc:
                    processed_doc["page_content"] = doc["chunk"]
                else:
                    # 내용이 없으면 기본값 설정하지만 결과에서 제외
                    logger.warning({
                        "event": "vector_search_empty_document",
                        "trace_id": trace_id,
                        "document_id": doc.get("doc_id", "unknown"),
                        "message": "문서 내용이 없는 결과 발견"
                    })
                    continue  # 내용이 없는 문서는 건너뛰기
                
                # 내용이 "문서 내용이 없습니다"인 경우 결과에서 제외
                if processed_doc["page_content"] == "문서 내용이 없습니다." or not processed_doc["page_content"].strip():
                    logger.warning({
                        "event": "vector_search_empty_content",
                        "trace_id": trace_id,
                        "document_id": doc.get("doc_id", "unknown"),
                        "message": "문서 내용이 빈 문자열이거나 '문서 내용이 없습니다'인 결과 제외"
                    })
                    continue  # 이런 문서도 건너뛰기
                
                # API 명세에 맞게 메타데이터 처리
                if "metadata" in doc:
                    processed_doc["metadata"] = doc["metadata"]
                else:
                    # 기존 응답 형식 지원
                    processed_doc["metadata"] = {
                        "title": doc.get("title", "제목 없음"),
                        "source": doc.get("source", "출처 불명"),
                        "author": doc.get("author", ""),
                        "date": doc.get("date", "")
                    }
                
                # API 명세에 맞게 score 처리
                if "score" in doc:
                    processed_doc["score"] = doc["score"]
                else:
                    processed_doc["score"] = 0.0
                
                results.append(processed_doc)
            
            # 내용 기반 필터링 후 결과가 없으면 로그 남기기
            if not results:
                logger.warning({
                    "event": "vector_search_no_valid_results",
                    "trace_id": trace_id,
                    "message": "유효한 내용이 있는 문서를 찾지 못했습니다. 필터링 전 문서 수: " + str(len(raw_results))
                })
            
            # 메트릭 기록
            team5_vector_search_duration.labels(service=service_name).observe(elapsed)
            team5_vector_search_results.labels(service=service_name).observe(len(results))
            
            logger.info({
                "event": "vector_search_success",
                "trace_id": trace_id,
                "elapsed_time": elapsed,
                "num_results": len(results),
                "processed_results_sample": str(results[:1])[:200] if results else "[]"
            })
            return results, elapsed
            
    except httpx.HTTPStatusError as e:
        elapsed = time.time() - start
        error_type = f"http_{e.response.status_code}"
        team5_vector_search_errors.labels(service=service_name, error_type=error_type).inc()
        team5_vector_search_duration.labels(service=service_name).observe(elapsed)
        
        response_text = "No response text available"
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            response_text = e.response.text
        
        logger.error({
            "event": "vector_search_http_error",
            "trace_id": trace_id,
            "error": str(e),
            "status_code": e.response.status_code if hasattr(e.response, 'status_code') else "unknown",
            "url": _API,
            "elapsed_time": elapsed,
            "response": response_text
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
            "url": _API,
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
            "error_type": error_type,
            "url": _API,
            "elapsed_time": elapsed
        })
        return [], elapsed