# src/evaluator.py
import logging

logger = logging.getLogger(__name__)
"""
청크별 요약·쿼리 생성, 검색 필요 판단, 문서 적합성 평가
"""

from typing import Dict, Any, List, TypedDict, Optional, Union

from src.llm import generate_summary, decide_need_search, decide_relevance
from src.config import get_settings

settings = get_settings()

# 타입 힌트를 위한 타입 정의
class NeedSearchResult(TypedDict):
    """LLM의 검색 필요성 판단 결과 상세 구조"""
    thought: str         # LLM의 판단 과정
    answer: str          # LLM의 원본 답변 (e.g., "Yes, ...", "No, ...")
    decision: bool       # 최종 검색 필요 여부 (True/False)
    success: bool        # LLM 호출 성공 여부
    error: Optional[str] # 오류 메시지 (실패 시)

class RelevanceResult(TypedDict):
    """문서 관련성 평가 결과 타입"""
    thought: str           # 평가 근거
    answer: str            # "Yes" 또는 "No"
    relevant: bool         # 관련성 여부
    feedback: str          # 재시도 시 개선을 위한 피드백
    retry_needed: bool     # 재시도 필요 여부
    success: bool          # 평가 성공 여부
    error: Optional[str]   # 오류 메시지 (실패 시)

class SummaryQueryResult(TypedDict):
    """요약 및 쿼리 생성 결과 타입"""
    summary_answer: str     # 문장 형식의 요약
    query: list[str]        # 검색에 사용할 키워드 리스트
    success: bool           # 요약 생성 성공 여부
    error: Optional[str]    # 오류 메시지 (실패 시)

def make_summary_and_query(chunk: str) -> SummaryQueryResult:
    """
    청크 텍스트로부터 요약을 생성하고, 이를 기반으로 검색 쿼리를 생성합니다.
    
    Args:
        chunk: 원본 청크 텍스트
        
    Returns:
        SummaryQueryResult: {
            "summary_answer": str,    # 문장 형식의 요약
            "query": list[str],      # 검색에 사용할 키워드 리스트 (요약에서 추출)
            "success": bool,         # 요약 생성 성공 여부
            "error": Optional[str]   # 오류 메시지 (실패 시)
        }
    """
    try:
        # 1. 문장 형식의 요약 생성
        summary = generate_summary(chunk)
        
        # 2. 요약에서 검색 쿼리 생성 (요약 자체를 쿼리로 사용)
        # 간단한 전처리: 문장 부호 제거, 중복 단어 제거, 불용어 제거
        import re
        
        # 문장 부호 제거 및 소문자 변환
        cleaned_summary = re.sub(r'[^\w\s]', ' ', summary).lower()
        
        # 불용어 목록 (필요에 따라 추가)
        stop_words = {"이", "그", "저", "것", "수", "등", "및", "또는", "그리고", 
                     "에서", "으로", "의", "을", "를", "은", "는", "이", "가", "에", "와", "과",
                     "도", "만", "에서", "에게", "께서", "에서부터", "까지", "처럼", "같이",
                     "처럼", "만큼", "만치", "보고", "위해", "때문에", "통해", "의해", "로서",
                     "로써", "으로서", "으로써", "이라고", "라는", "하고", "이랑", "이란"}
        
        # 단어 분리 및 불용어 제거
        words = [word for word in cleaned_summary.split() if word not in stop_words]
        
        # 중복 제거 (순서 유지)
        seen = set()
        keywords = [word for word in words if not (word in seen or seen.add(word))]
        
        # 상위 5개 키워드 선택 (없으면 전체 사용)
        max_keywords = 5
        query_keywords = keywords[:max_keywords] if len(keywords) > max_keywords else keywords
        
        return {
            "summary_answer": summary,
            "query": query_keywords,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"요약 및 쿼리 생성 중 오류 발생: {str(e)}")
        return {
            "summary_answer": "",
            "query": [],
            "success": False,
            "error": f"요약 및 쿼리 생성 중 오류가 발생했습니다: {str(e)}"
        }

def need_search(chunk: str, summary: str) -> NeedSearchResult:
    """
    LLM을 호출하여 검색 필요성 여부를 판단하고, LLM의 상세 응답을 반환합니다.
    
    Args:
        chunk: 원본 청크 텍스트
        summary: 청크의 요약
        
    Returns:
        NeedSearchResult: LLM의 검색 필요성 판단 결과 상세.
                          포함 필드: 'thought', 'answer', 'decision' (bool), 
                                     'success', 'error'.
    """
    # llm.decide_need_search는 이미 {'thought': ..., 'answer': ..., 'decision': True/False, 'success': ..., 'error': ...} 형식으로 반환합니다.
    return decide_need_search(chunk, summary)

def evaluate_relevance(chunk: str, summary: str, doc: dict[str, Any], retry_count: int = 0) -> RelevanceResult:
    """
    문서의 관련성을 평가하고, 필요한 경우 재시도 로직을 적용합니다.
    
    Args:
        chunk: 원본 청크 텍스트
        summary: 청크의 요약
        doc: 평가할 문서 (page_content와 metadata 포함)
        retry_count: 재시도 횟수 (기본값: 0)
        
    Returns:
        RelevanceResult: 문서 관련성 평가 결과
    """
    try:
        # 문서 관련성 평가
        result = decide_relevance(chunk, summary, doc)
        
        # 재시도 로직
        if result.get("retry_needed", False) and retry_count < settings.MAX_RETRIES:
            logger.info(f"문서 관련성 부족으로 재시도 (시도: {retry_count + 1})")
            return evaluate_relevance(chunk, summary, doc, retry_count + 1)
            
        return {
            "thought": result.get("thought", ""),
            "answer": result.get("answer", "No"),
            "relevant": result.get("relevant", False),
            "feedback": result.get("feedback", ""),
            "retry_needed": result.get("retry_needed", False),
            "success": True,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"문서 관련성 평가 중 오류: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "thought": error_msg,
            "answer": "No",
            "relevant": False,
            "feedback": "문서 평가 중 오류가 발생했습니다. 다시 시도해주세요.",
            "retry_needed": True,
            "success": False,
            "error": error_msg
        }
