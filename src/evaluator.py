# src/evaluator.py
import logging
import uuid

logger = logging.getLogger(__name__)
"""
청크별 요약·쿼리 생성, 검색 필요 판단, 문서 적합성 평가
"""

from typing import Dict, Any, List, TypedDict, Optional, Union, Tuple, cast

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
    trace_id = str(uuid.uuid4())
    logger.info({
        "event": "make_summary_query_start",
        "trace_id": trace_id,
        "chunk_length": len(chunk),
        "chunk_preview": chunk[:100] + ("..." if len(chunk) > 100 else "")
    })
    
    try:
        # 1. 문장 형식의 요약 생성 - 이제 예외가 발생하면 바로 전파됨
        summary = generate_summary(chunk)
        
        # 2. 요약에서 검색 쿼리 추출
        import re
        
        # 문장 부호 제거 및 소문자 변환
        cleaned_summary = re.sub(r'[^\w\s]', ' ', summary).lower()
        
        # 한국어 또는 영어인지 확인
        is_korean = bool(re.search(r'[가-힣]', summary))
        
        # 불용어 목록 (한국어/영어에 따라 다름)
        if is_korean:
            stop_words = {"이", "그", "저", "것", "수", "등", "및", "또는", "그리고", 
                        "에서", "으로", "의", "을", "를", "은", "는", "이", "가", "에", "와", "과",
                        "도", "만", "에서", "에게", "께서", "에서부터", "까지", "처럼", "같이",
                        "처럼", "만큼", "만치", "보고", "위해", "때문에", "통해", "의해", "로서",
                        "로써", "으로서", "으로써", "이라고", "라는", "하고", "이랑", "이란"}
        else:
            stop_words = {"a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
                        "with", "by", "about", "as", "from", "like", "since", "this", "that", "these",
                        "those", "it", "is", "was", "were", "be", "been", "has", "have", "had"}
        
        # 단어 분리
        if is_korean:
            # 한국어는 띄어쓰기가 불규칙할 수 있으므로 2-3자 이상의 단어만 추출
            words = [word for word in cleaned_summary.split() if len(word) >= 2 and word not in stop_words]
        else:
            # 영어는 일반적인 단어 분리
            words = [word for word in cleaned_summary.split() if word not in stop_words]
        
        # 핵심 키워드 선택을 위한 가중치 계산 (간단한 TF 적용)
        word_counts = {}
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # 가중치 순으로 정렬
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 키워드 선택 (최대 10개)
        max_keywords = 10
        keywords = [word for word, count in sorted_words[:max_keywords]]
        
        # 키워드가 너무 적으면 원래 단어 목록에서 채움
        if len(keywords) < 5 and len(words) > len(keywords):
            # 중복 제거 (순서 유지)
            seen = set(keywords)
            for word in words:
                if word not in seen and len(keywords) < 5:
                    keywords.append(word)
                    seen.add(word)
        
        logger.info({
            "event": "make_summary_query_success",
            "trace_id": trace_id,
            "summary": summary,
            "keywords": keywords,
            "keyword_count": len(keywords)
        })
        
        return {
            "summary_answer": summary,
            "query": keywords,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"요약 및 쿼리 생성 중 오류 발생: {str(e)}"
        logger.error({
            "event": "make_summary_query_error",
            "trace_id": trace_id,
            "error": error_msg,
            "error_type": type(e).__name__
        }, exc_info=True)
        
        # 요약 생성 중 오류가 발생하면 예외를 상위로 전파
        raise RuntimeError(error_msg) from e

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

def evaluate_documents(
    chunk: str,
    summary: str,
    documents: List[Dict[str, Any]],
    retry_count: int = 0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    """
    검색된 문서를 평가하여 관련성 여부를 판단합니다.
    
    Args:
        chunk: 원본 청크 텍스트
        summary: 청크 요약
        documents: 검색된 문서 리스트
        retry_count: 현재 재시도 횟수
        
    Returns:
        Tuple[
            List[Dict[str, Any]]: 관련 있는 문서 리스트
            Dict[str, Any]: 평가 결과 메타데이터
            bool: 재시도 필요 여부
        ]
    """
    logger.info(f"문서 평가 시작. 평가할 문서 수: {len(documents)}")
    
    # 검색 결과가 없으면 빈 결과와 재시도 필요 반환
    if not documents:
        return [], {"feedback": "검색 결과가 없습니다. 다른 쿼리로 시도해보세요."}, True
    
    relevant_docs = []
    retry_needed = False
    final_feedback = ""
    max_retries = settings.MAX_RETRIES  # settings에서 MAX_RETRIES를 가져옴
    
    try:
        for doc in documents:
            try:
                # LLM으로 문서 관련성 평가
                eval_result = decide_relevance(chunk, summary, doc)
                
                # 관련 문서인 경우 추가
                if eval_result["relevant"]:
                    relevant_docs.append(doc)
                
                # 재시도 필요 여부 확인 (피드백 기반)
                if eval_result["retry_needed"]:
                    retry_needed = True
                    final_feedback = eval_result.get("feedback", "")  # 피드백 저장
                
            except Exception as e:
                logger.error(f"문서 관련성 평가 중 오류: {str(e)}")
                # 오류 발생 시 재시도 필요로 표시
                retry_needed = True
                final_feedback = f"문서 평가 중 오류가 발생했습니다. 다시 시도해주세요. 오류: {str(e)}"
    
        # 결과 요약
        if retry_count >= max_retries:
            retry_needed = False  # 최대 재시도 횟수 초과 시 더 이상 재시도하지 않음
            logger.info(f"최대 재시도 횟수({max_retries})에 도달했습니다. 더 이상 재시도하지 않습니다.")
        
        # 관련 문서가 없고, 검색 결과가 있으며, 재시도 가능한 경우
        should_retry = (len(relevant_docs) == 0 and len(documents) > 0 and retry_count < max_retries)
        
        metadata = {
            "evaluated": len(documents),
            "relevant": len(relevant_docs),
            "feedback": final_feedback or "관련 문서를 찾지 못했습니다. 다른 키워드로 검색해보세요.",
            "retry_count": retry_count,
            "max_retries": max_retries
        }
        
        logger.info(f"[evaluate_documents] 청크 ID {retry_count} (인덱스 {retry_count}): 평가 완료. 관련 문서 수: {len(relevant_docs)}/{len(documents)}. 재시도 최종 결정: {should_retry} (근거: 관련문서 없음? {len(relevant_docs) == 0}, 검색된문서 있음? {len(documents) > 0}, 재시도 가능횟수 남음? {retry_count < max_retries}). 피드백 (앞 50자): {final_feedback[:50]}...")
        
        return relevant_docs, metadata, should_retry
        
    except Exception as e:
        logger.error(f"전체 문서 평가 과정 중 오류: {str(e)}")
        return [], {"feedback": f"문서 평가 중 오류가 발생했습니다: {str(e)}"}, True
