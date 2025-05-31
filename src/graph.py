# src/graph.py 상단 부분 - 메트릭 임포트 수정

import time
import logging
import uuid
from .logging_utils import setup_json_logger
from typing import Any, Dict, List, TypedDict, Optional, Callable, Union, Literal, Annotated
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Send

from src.chunker import chunk_sentences
from src.evaluator import make_summary_and_query, need_search, evaluate_documents
from src.vector_search import search_vector
from src.config import get_settings

# JSON Logger 설정
setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")
settings = get_settings()

# 모든 필요한 메트릭 임포트 (추가된 부분)
from .metrics import (
    team5_llm_calls,
    team5_llm_duration, 
    team5_llm_errors,
    team5_llm_tokens,
    team5_pipeline_executions,
    team5_pipeline_duration,
    team5_pipeline_errors,
    team5_pipeline_active_executions
)
def log_execution_time(func: Callable) -> Callable:
    """함수 실행 시간을 측정하는 데코레이터"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        stage_name = func.__name__
        service_name = "server2-rag"
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # 스테이지별 에러 메트릭 기록
            error_type = type(e).__name__
            team5_pipeline_errors.labels(service=service_name, stage=stage_name, error_type=error_type).inc()
            raise
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} 실행 시간: {elapsed:.2f}초")
    return wrapper
# 상태 정의
class GraphState(TypedDict):
    """워크플로우 상태를 나타내는 타입 힌트"""
    text: str
    chunk: str
    chunks: List[str]
    current_chunk_index: int
    total_chunks: int  # 총 청크 수
    results: List[Dict[str, Any]]
    has_next: bool
    should_retry: bool
    retry_count: int   # 현재 청크의 재시도 횟수
    feedback: str
    entry: Dict[str, Any]
    summary: Dict[str, Any]
    decision: Dict[str, Any]
    similar_documents: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    error: Optional[str] = None

# 노드 정의
def create_chunks(state: GraphState) -> Dict[str, Any]:
    """텍스트를 청크로 분할하고 초기 상태를 설정합니다."""
    start_time = time.time()
    service_name = "server2-rag"
    node_name = "create_chunks"
    
    try:
        text_input = state.get("text", "").strip()
        logger.info(f"텍스트 청킹 시작. 원본 텍스트 길이: {len(text_input)}자")

        if not text_input:
            logger.warning("입력된 텍스트가 비어 있습니다.")
            result = {
                **state, # 이전 상태 유지
                "text": text_input,
                "chunks": [],
                "total_chunks": 0,
                "current_chunk_index": 0,
                "results": [],
                "has_next": False,
                "error": "입력 텍스트 없음"
            }
            # 노드 실행 실패 기록
            from src.metrics import track_node_execution
            track_node_execution(service_name, node_name, time.time() - start_time, success=False)
            return result

        chunk_list = chunk_sentences(text_input)
        logger.info(f"[create_chunks] 총 {len(chunk_list)}개의 청크 생성됨.")
        
        for i, c in enumerate(chunk_list):
            logger.debug(f"청크 {i+1} (길이: {len(c)}자): {c[:100]}...")

        result = {
            **state, # 이전 상태 유지
            "text": text_input,
            "chunks": chunk_list,
            "total_chunks": len(chunk_list),
            "current_chunk_index": 0,
            "results": [],
            "has_next": len(chunk_list) > 0,
            # 다음 process_chunk에서 설정될 값들 초기화
            "chunk": "",
            "retry_count": 0,
            "should_retry": False,
            "feedback": "",
            "entry": {},
            "summary": {},
            "decision": {},
            "similar_documents": [],
            "error": None
        }
        
        # 노드 실행 성공 기록
        from src.metrics import track_node_execution
        track_node_execution(service_name, node_name, time.time() - start_time, success=True)
        return result
        
    except Exception as e:
        error_msg = f"청크 생성 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # 노드 실행 실패 기록
        from src.metrics import track_node_execution
        track_node_execution(service_name, node_name, time.time() - start_time, success=False)
        
        return {
            **state, # 이전 상태 유지
            "text": text_input if 'text_input' in locals() else "",
            "chunks": [],
            "total_chunks": 0,
            "current_chunk_index": 0,
            "results": [],
            "has_next": False,
            "error": error_msg
        }

def process_chunk(state: GraphState) -> Dict[str, Any]:
    """현재 인덱스의 청크를 설정하고 재시도 관련 상태를 초기화합니다."""
    chunks_list = state.get("chunks", [])
    current_idx = state.get("current_chunk_index", 0)
    total_chunks_count = state.get("total_chunks", 0)

    logger.info(f"청크 처리 시작. 현재 인덱스: {current_idx + 1}/{total_chunks_count}")

    if current_idx >= total_chunks_count:
        logger.info("모든 청크 처리 완료 또는 인덱스 오류.")
        return {
            **state,
            "chunk": "",
            "has_next": False, # 더 이상 진행할 청크 없음
            "error": "청크 인덱스 초과" if total_chunks_count > 0 else None
        }

    current_chunk_text = chunks_list[current_idx]
    logger.info(f"[process_chunk] 현재 청크 인덱스: {current_idx}, 총 청크 수: {state.get('total_chunks')}. 청크 내용 (앞 50자): {current_chunk_text[:50]}...")
    logger.debug(f"처리 중인 청크 {current_idx + 1} (길이: {len(current_chunk_text)}자): {current_chunk_text[:100]}...")

    # 새 청크 또는 새 재시도 시, 이전 요약/결정/문서 상태 초기화
    return {
        **state,
        "chunk": current_chunk_text,
        "retry_count": 0,  # 새 청크 처리 시작 시 재시도 횟수 초기화
        "should_retry": False,
        "feedback": "",
        "entry": {
            "start_time": time.time(),
            "chunk_id": current_idx,
            "total_chunks": total_chunks_count,
            "original_chunk_text": current_chunk_text # 재시도 시 원본 청크 유지용
        },
        "summary": {},
        "decision": {},
        "similar_documents": [],
        "error": None
    }

def generate_summary(state: GraphState) -> Dict[str, Any]:
    """요약 및 쿼리 생성"""
    chunk = state.get("chunk", "")
    if not chunk:
        logger.warning("빈 청크가 전달되어 요약을 생성할 수 없습니다.")
        return {
            "chunk": "",
            "summary": {"thought": "", "answer": "", "query": ""},
            "entry": state.get("entry", {"start_time": time.time()}),
            "has_next": False
        }
    
    # evaluator의 make_summary_and_query 호출
    logger.info(f"[generate_summary] 청크 (앞 50자): {chunk[:50]}... 에 대한 요약 생성 시작")
    
    try:
        # 이제 make_summary_and_query 함수는 예외 발생 시 상위로 전파함
        summ = make_summary_and_query(chunk)
        
        # 요약 로깅
        summary_text = summ.get("summary_answer", "")
        query_keywords = summ.get("query", [])
        
        logger.info({
            "event": "summary_generated",
            "chunk_id": state.get("entry", {}).get("chunk_id", "unknown"),
            "summary": summary_text,
            "summary_length": len(summary_text),
            "query_keywords": query_keywords,
            "keyword_count": len(query_keywords)
        })
        
        return {
            "chunk": chunk,
            "chunks": state.get("chunks", []),
            "current_chunk_index": state.get("current_chunk_index", 0),
            "results": state.get("results", []),
            "has_next": state.get("has_next", False),
            "summary": {
                "thought": summ.get("thought", ""),
                "answer": summ.get("summary_answer", ""),
                "query": summ.get("query", [])
            },
            "entry": {
                **state.get("entry", {"start_time": time.time()}),
                "summary_thought": summ.get("thought", ""),
                "summary_answer": summ.get("summary_answer", "")
            }
        }
    except Exception as e:
        # 오류 발생 시 시스템 중단을 위해 예외를 다시 발생시킴
        error_msg = f"요약 생성 중 치명적 오류 발생: {str(e)}"
        logger.error({
            "event": "summary_generation_critical_error",
            "chunk_id": state.get("entry", {}).get("chunk_id", "unknown"),
            "error": error_msg,
            "error_type": type(e).__name__
        }, exc_info=True)
        
        # 오류를 다시 발생시켜 시스템을 중단시킴
        raise RuntimeError(error_msg) from e

def decide_search(state: GraphState) -> Dict[str, Any]:
    """검색 필요 여부 결정"""
    chunk = state.get("chunk", "")
    summary_data = state.get("summary", {}) # summary는 딕셔너리임
    summary_answer = summary_data.get("answer", "") # summary_answer는 문자열

    # 공통 반환 필드 준비
    common_state_updates = {
        "chunk": chunk,
        "chunks": state.get("chunks", []),
        "current_chunk_index": state.get("current_chunk_index", 0),
        "results": state.get("results", []),
        "has_next": state.get("has_next", False),
        "summary": summary_data, # 전체 summary 객체 전달
        "entry": state.get("entry", {"start_time": time.time()}),
        "error": state.get("error") # 이전 에러 상태 유지
    }

    if not chunk or not summary_answer:
        logger.warning("[decide_search] 청크 또는 요약 답변이 비어 있어 검색 필요 여부 판단을 건너뛰었습니다. 'decision'을 False로 설정합니다.")
        return {
            **common_state_updates,
            "decision": {"thought": "입력 부족", "answer": "No", "decision": False, "success": True, "error": None}, # success: True, error: None 으로 명시
        }
    
    # need_search는 llm.decide_need_search를 호출하며, 
    # 반환 값: {"thought": ..., "answer": ..., "decision": ..., "success": ..., "error": ...}
    decision_result = need_search(chunk, summary_answer) 
    
    logger.info(f"[decide_search] llm.decide_need_search 결과: {decision_result}")

    # LLM 호출 성공 여부 확인
    if not decision_result.get("success", False):
        error_message = decision_result.get("error", "LLM 호출 실패 (원인 불명)")
        logger.error(f"[decide_search] llm.decide_need_search 호출 실패: {error_message}")
        # 상태에 에러 기록하고, decision은 False로 설정하여 검색 없이 진행
        return {
            **common_state_updates,
            "decision": {
                "thought": decision_result.get("thought", "LLM 호출 실패"),
                "answer": decision_result.get("answer", "No"), # 기본값 "No"
                "decision": False, # 검색 안 함으로 결정
                "success": False,
                "error": error_message
            },
            "error": f"decide_search: {error_message}" # GraphState의 최상위 error 필드에도 반영
        }

    # LLM 호출 성공 시
    actual_decision_bool = decision_result.get("decision", False)
    logger.info(f"[decide_search] 추출된 실제 결정 (boolean): {actual_decision_bool}")

    # entry 필드 업데이트 (기존 entry 내용 유지하며 추가)
    updated_entry = {
        **common_state_updates.get("entry", {}), # common_state_updates에 있는 entry를 가져옴
        "decide_thought": decision_result.get("thought", ""),
        "decide_answer": decision_result.get("answer", ""), # LLM의 원본 answer ("Yes" 또는 "No")
        "need_search_decision": actual_decision_bool # boolean decision 값
    }

    # llm.decide_need_search의 결과 (decision_result)는 다음과 같은 키를 가짐:
    # "thought", "answer" (LLM의 원본 텍스트 답변), "decision" (boolean),
    # "success" (boolean), "error" (메시지 또는 None)
    return {
        **common_state_updates,
        "decision": {
            "thought": decision_result.get("thought", ""),
            "answer": decision_result.get("answer", ""), # LLM의 원본 텍스트 답변
            "decision": actual_decision_bool, # decision_result.get("decision", False)와 동일
            "success": True, # 이 경로는 llm 호출 성공 케이스
            "error": None    # 성공 케이스이므로 에러 없음
        },
        "entry": updated_entry
        # "error"는 common_state_updates에서 가져온 전역 error 상태를 유지.
        # decide_need_search 자체의 성공/실패는 GraphState.decision.success/error로 확인.
    }

def search_documents(state: GraphState) -> Dict[str, Any]:
    """벡터 검색 수행"""
    if not state["decision"]["decision"]:
        return {"similar_documents": []}
        
    query = state["summary"]["query"]
    results, _ = search_vector(query, settings.VECTOR_TOP_K)
    
    return {"similar_documents": results}

def evaluate_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    검색된 문서 관련성 평가 및 재시도 결정
    """
    start_time = time.time()
    service_name = "server2-rag"
    node_name = "evaluate_relevance"
    
    chunk = state.get("chunk", "")
    summary = state.get("summary", {}).get("answer", "")
    documents = state.get("similar_documents", [])
    retry_count = state.get("retry_count", 0)
    
    logger.info(f"문서 평가 시작. 평가할 문서 수: {len(documents)}")
    
    try:
        # evaluator.py의 evaluate_documents 함수 호출
        relevant_docs, metadata, should_retry = evaluate_documents(
            chunk=chunk,
            summary=summary,
            documents=documents,
            retry_count=retry_count
        )
        
        # 평가 결과 로깅
        logger.info(f"[evaluate_documents] 청크 ID {state.get('entry', {}).get('chunk_id', 0)} (인덱스 {retry_count}): 평가 완료. 관련 문서 수: {len(relevant_docs)}/{len(documents)}. 재시도 최종 결정: {should_retry}. 피드백 (앞 50자): {metadata.get('feedback', '')[:50]}...")
        
        # 메트릭 기록 - 검색 관련성 통계
        from src.metrics import update_search_relevance_metrics
        update_search_relevance_metrics(service_name, len(relevant_docs), len(documents))
        
        # 재시도 추적
        from src.metrics import track_retry
        if should_retry:
            is_max_reached = retry_count >= settings.MAX_RETRIES - 1
            track_retry(service_name, "evaluate_documents", is_max_reached)
        
        # 노드 실행 성공 기록
        from src.metrics import track_node_execution
        track_node_execution(service_name, node_name, time.time() - start_time, success=True)
        
        return {
            **state,
            "similar_documents": relevant_docs,
            "should_retry": should_retry,
            "feedback": metadata.get("feedback", "관련 문서를 찾지 못했습니다.")
        }
    except Exception as e:
        error_msg = f"문서 평가 중 오류: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # 노드 실행 실패 기록
        from src.metrics import track_node_execution
        track_node_execution(service_name, node_name, time.time() - start_time, success=False)
        
        # 오류 발생 시 원래 문서 목록 유지하고 재시도 플래그 설정
        return {
            **state,
            "should_retry": True,
            "feedback": f"문서 평가 중 오류 발생: {str(e)}",
            "error": error_msg
        }

def update_results(state: GraphState) -> Dict[str, Any]:
    """
    결과 업데이트 및 다음 처리 단계 결정
    
    재시도가 필요한 경우 요약/쿼리 생성 단계로 돌아가고,
    그렇지 않으면 다음 청크로 넘어갑니다.
    
    Returns:
        Dict[str, Any]: 업데이트된 상태 정보
    """
    try:
        # 현재 상태에서 필요한 정보 추출
        current_chunk = state.get("chunk", "")
        current_index = state.get("current_chunk_index", 0) # process_chunk에서 설정한 인덱스
        total_chunks = state.get("total_chunks", 0)
        chunks = state.get("chunks", [])
        results = state.get("results", [])
        entry = state.get("entry", {})
        should_retry_flag = state.get("should_retry", False)
        current_retry_count_from_state = state.get("retry_count", 0)
        logger.info(f"[update_results] 청크 ID {entry.get('chunk_id','N/A')} (인덱스 {current_index}): 결과 업데이트 시작. 재시도 플래그: {should_retry_flag}, 현재 재시도 카운트(상태): {current_retry_count_from_state}")
        
        # 평가 결과에서 관련 문서 추출
        relevant_docs = state.get("similar_documents", [])
        should_retry = state.get("should_retry", False)
        feedback = state.get("feedback", "")
        
        # 현재 청크에 대한 결과 생성 
        # 요구된 형식에 맞게 chunk 필드로 통일
        result_entry = {
            "chunk": current_chunk,  # 영어로 번역된 값이 필요할 경우 이 부분을 추가 처리
            "summary": state.get("summary", {}).get("answer", ""),
            "similar_documents": [
                {
                    "page_content": doc.get("page_content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0)
                }
                for doc in relevant_docs
            ],
            "elapsed_time": time.time() - entry.get("start_time", time.time()),
            "error": state.get("error")
        }
        
        # 결과 목록에 추가
        new_results = results + [result_entry]
        
        # 다음 청크 처리 여부 확인
        has_next = current_index < len(chunks) - 1
        
        # 재시도가 필요한 경우 요약        # 재시도가 필요한 경우
        if should_retry_flag:
            new_retry_count = current_retry_count_from_state + 1
            logger.info(f"[update_results] 청크 ID {entry.get('chunk_id','N/A')} (인덱스 {current_index}): 'should_retry'가 True. 재시도 결정. 재시도 횟수 {new_retry_count}로 증가. generate_summary로 이동.")
            return {
                **state, # 현재 상태 대부분 유지
                "results": results, # 재시도이므로 현재 청크 결과는 아직 results에 반영 안 함
                "retry_count": state.get("retry_count", 0) + 1,
                "should_retry": True, # 명시적으로 재시도 플래그 설정
                "feedback": feedback, # LLM에 전달할 피드백
                # current_chunk_index와 has_next는 변경하지 않고 현재 청크 재시도
            }
        
        # 정상 완료 또는 최대 재시도 도달로 재시도 안 함
        logger.info(f"[update_results] 청크 ID {entry.get('chunk_id','N/A')} (인덱스 {current_index}): 'should_retry'가 False. 현재 청크 결과 저장.")
        updated_results = results + [result_entry] # 현재 청크 결과를 전체 결과에 추가
        
        next_chunk_index = current_index + 1
        has_next_chunk = next_chunk_index < total_chunks
        if has_next_chunk:
            logger.info(f"[update_results] 청크 ID {entry.get('chunk_id','N/A')} (인덱스 {current_index}): 다음 청크 {next_chunk_index} 처리 위해 process_chunk로 이동.")
        else:
            logger.info(f"[update_results] 청크 ID {entry.get('chunk_id','N/A')} (인덱스 {current_index}): 모든 청크 ({current_index + 1}/{total_chunks}) 처리 완료. finalize_results로 이동.")

        return {
            **state,
            "results": updated_results,
            "current_chunk_index": next_chunk_index, # 다음 청크로 인덱스 이동
            "has_next": has_next_chunk,
            "should_retry": False, # 재시도 플래그 해제
            "retry_count": 0, # 다음 청크를 위해 재시도 횟수 초기화 (process_chunk에서도 하지만 여기서도 명시)
            "feedback": "",
            "error": None # 현재 청크 성공적으로 처리
        }
        
    except Exception as e:
        error_msg = f"결과 업데이트 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "error": error_msg,
            "has_next": False,
            "should_retry": False
        }

def finalize_results(state: GraphState) -> Dict[str, Any]:
    """
    최종 결과를 정리하고 워크플로우 실행 통계를 반환합니다.
    
    Args:
        state: 현재 워크플로우 상태
        
    Returns:
        Dict[str, Any]: {
            "result": 처리된 청크 결과 목록,
            "total_elapsed_time": 전체 실행 시간(초)
        }
    """
    try:
        start_time = state.get("entry", {}).get("start_time", time.time())
        total_elapsed = time.time() - start_time
        
        logger.info(f"파이프라인 처리 완료. 총 실행 시간: {total_elapsed:.2f}초")
        
        # 결과 반환: LangGraph가 요구하는 'final_result' 키 사용
        return {
            "final_result": {
                "result": state.get("results", []),
                "total_elapsed_time": total_elapsed
            }
        }
        
    except Exception as e:
        error_msg = f"최종 결과 정리 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "final_result": {
                "result": [],
                "error": error_msg,
                "total_elapsed_time": time.time() - start_time if 'start_time' in locals() else 0
            }
        }

from langgraph.checkpoint.memory import MemorySaver # 추가된 임포트

# 워크플로우 정의
def create_rag_workflow() -> CompiledGraph:
    """
    RAG 워크플로우를 생성하고 컴파일합니다.
    
    Returns:
        CompiledGraph: 컴파일된 워크플로우 그래프
    """
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("create_chunks", create_chunks)
    workflow.add_node("process_chunk", process_chunk)
    workflow.add_node("generate_summary", generate_summary)
    workflow.add_node("decide_search", decide_search)  # 검색 결정 노드 추가
    workflow.add_node("search_documents", search_documents)
    workflow.add_node("evaluate_relevance", evaluate_documents_node)  # 함수명 수정 (evaluate_relevance -> evaluate_documents_node)
    workflow.add_node("update_results", update_results)
    workflow.add_node("finalize_results", finalize_results)
    
    # 엣지 설정
    workflow.set_entry_point("create_chunks")
    
    # 1. 최초 진입: create_chunks
    workflow.set_entry_point("create_chunks")

    # 2. create_chunks -> 첫 청크 처리 또는 종료
    def after_create_chunks(state: GraphState) -> str:
        service_name = "server2-rag"
        from_node = "create_chunks"
        to_node = ""
        
        if state.get("error"): # 청킹 실패
            logger.error(f"청킹 실패로 파이프라인 종료: {state.get('error')}")
            to_node = "finalize_results"
        elif not state.get("has_next", False): # 청크 없음
            logger.info("처리할 청크가 없어 파이프라인 종료.")
            to_node = "finalize_results"
        else:
            to_node = "process_chunk" # 청크 있으면 첫 청크 처리
            
        # 노드 전환 추적
        from src.metrics import track_node_transition
        track_node_transition(service_name, from_node, to_node)
        
        return to_node

    workflow.add_conditional_edges(
        "create_chunks",
        after_create_chunks
    )

    # 3. process_chunk -> 요약 또는 종료 (오류 시)
    def after_process_chunk(state: GraphState) -> str:
        service_name = "server2-rag"
        from_node = "process_chunk"
        to_node = ""
        
        if state.get("error"): # 청크 처리 중 오류 (예: 인덱스 초과)
            logger.error(f"청크 처리 중 오류로 파이프라인 종료: {state.get('error')}")
            to_node = "finalize_results"
        else:
            # process_chunk에서 has_next가 False로 설정되면 여기서 잡히지 않음.
            # 대신, update_results 이후 분기에서 처리.
            to_node = "generate_summary"
            
        # 노드 전환 추적
        from src.metrics import track_node_transition
        track_node_transition(service_name, from_node, to_node)
        
        return to_node

    workflow.add_conditional_edges(
        "process_chunk",
        after_process_chunk
    )
    
    # 4. generate_summary -> decide_search
    workflow.add_edge("generate_summary", "decide_search")
    
    # 5. decide_search -> search_documents 또는 update_results (검색 불필요)
    def decide_search_flow(state: GraphState) -> str:
        service_name = "server2-rag"
        from_node = "decide_search"
        to_node = ""
        
        # decide_search 노드에서 이미 decision 결과를 상태에 저장했으므로, 해당 값을 사용합니다.
        # decision_dict는 {"thought": ..., "answer": ..., "decision": ..., "success": ..., "error": ...} 형태입니다.
        decision_dict = state.get("decision", {})
        
        # LLM 호출이 성공했고, 검색이 필요하다고 결정된 경우
        search_needed = decision_dict.get("success", False) and decision_dict.get("decision", False)
        
        # 로그 추가: decision_dict 전체와 최종 search_needed 값
        logger.info(f"[decide_search_flow] GraphState.decision 값: {decision_dict}")
        logger.info(f"[decide_search_flow] 계산된 search_needed: {search_needed}")

        if search_needed:
            logger.info("[decide_search_flow] 검색 필요: 'search_documents'로 이동합니다.")
            to_node = "search_documents"
        else:
            # 검색이 필요 없거나, decide_search 단계에서 오류가 발생한 경우 (success=False)
            if not decision_dict.get("success", False):
                logger.warning("[decide_search_flow] 검색 결정 단계에서 오류 발생. 'update_results'로 이동합니다.")
            else:
                logger.info("[decide_search_flow] 검색 불필요: 'update_results'로 이동합니다.")
            to_node = "update_results"
            
        # 노드 전환 추적
        from src.metrics import track_node_transition
        track_node_transition(service_name, from_node, to_node)
        
        return to_node
    
    workflow.add_conditional_edges(
        "decide_search",
        decide_search_flow
    )
    
    # 6. search_documents -> evaluate_relevance
    workflow.add_edge("search_documents", "evaluate_relevance")
    
    # 7. evaluate_relevance -> update_results
    workflow.add_edge("evaluate_relevance", "update_results")
    
    # 8. update_results -> 재시도 또는 다음 청크 또는 종료
    def after_update_results(state: GraphState) -> str:
        service_name = "server2-rag"
        from_node = "update_results"
        to_node = ""
        
        if state.get("should_retry", False):
            to_node = "generate_summary"
            logger.info(f"청크 {state.get('entry',{}).get('chunk_id','N/A') + 1} 재시도: generate_summary로 이동")
        elif state.get("has_next", False):
            next_idx_to_process = state.get('current_chunk_index', 0)
            processed_idx = next_idx_to_process - 1 # The index that was just processed
            logger.info(f"[after_update_results] 이전 청크 (인덱스 {processed_idx}) 처리 완료. 'process_chunk'로 라우팅하여 다음 청크 (인덱스 {next_idx_to_process})를 처리합니다.")
            to_node = "process_chunk"
        else:
            total_chunks_count = state.get('total_chunks', 0)
            logger.info(f"[after_update_results] 모든 청크 처리 완료 ({total_chunks_count}/{total_chunks_count}). 'finalize_results'로 라우팅합니다.")
            to_node = "finalize_results"
        
        # 노드 전환 추적
        from src.metrics import track_node_transition
        track_node_transition(service_name, from_node, to_node)
        
        return to_node
    
    workflow.add_conditional_edges(
        "update_results",
        after_update_results
    )
    
    # 9. finalize_results -> END
    workflow.add_edge("finalize_results", END)

    # 워크플로우 컴파일
    logger.info("RAG 워크플로우 컴파일 중...")
    memory = MemorySaver() # MemorySaver 인스턴스 생성
    compiled_workflow = workflow.compile(checkpointer=memory)
    logger.info("RAG 워크플로우 컴파일 완료.")
    return compiled_workflow

# 전역 워크플로우 인스턴스
rag_workflow = create_rag_workflow()

def run_pipeline(text: str, thread_id: str = None) -> Dict[str, Any]:
    """
    RAG 파이프라인 실행
    
    Args:
        text: 처리할 텍스트
        thread_id: (선택 사항) 대화 스레드 ID
        
    Returns:
        Dict[str, Any]: 처리 결과
    """
    logger.info(f"[run_pipeline] 함수 시작. 입력 텍스트 길이: {len(text)}자, 스레드 ID: {thread_id}")
    start_time = time.time()
    service_name = "server2-rag"
    
    # 파이프라인 실행 메트릭 증가
    team5_pipeline_executions.labels(service=service_name).inc()
    team5_pipeline_active_executions.labels(service=service_name).inc()
    
    try:
        logger.info("파이프라인 실행 시작")
        
        # 초기 상태 설정 - 모든 필수 필드 포함
        initial_state = {
            # 기본 필드
            "text": text,
            "chunks": [],
            "current_chunk_index": 0,
            "total_chunks": 0,
            "results": [], # 개별 청크 결과 누적
            "has_next": False,
            "should_retry": False,
            "retry_count": 0,
            "feedback": "",
            "summary": {},
            "decision": {},
            "similar_documents": [],
            "entry": {"start_time": start_time}, # finalize_results에서 사용하기 위해 시작 시간 추가
            "final_result": {}, # finalize_results 노드가 채울 필드
            "error": None
        }
        
        logger.debug(f"초기 상태 설정 완료: {initial_state}")
        
        logger.info(f"워크플로우 실행 시작. Thread ID: {thread_id}")
        config = {
            "recursion_limit": settings.LANGRAPH_RECURSION_LIMIT,  # 설정 파일에서 값 가져오기
            "configurable": {"thread_id": thread_id}
        }
        
        # rag_workflow.invoke는 최종 상태(final_state)를 반환합니다.
        final_state = rag_workflow.invoke(initial_state, config=config)
        logger.info(f"워크플로우 실행 완료. 최종 상태 키: {list(final_state.keys())}")

        # finalize_results 노드에서 설정한 final_result 값을 가져옴
        pipeline_outcome = final_state.get("final_result", {})
        
        api_result = pipeline_outcome.get("result", [])
        api_total_elapsed_time = pipeline_outcome.get("total_elapsed_time", 0)
        pipeline_error = pipeline_outcome.get("error")

        # 결과 형식 수정 - chunk 키 사용 (chunk_en 대신)
        for chunk_result in api_result:
            # chunk 필드 확인 및 설정
            if "chunk" not in chunk_result:
                # 원본 텍스트를 chunk 필드로 설정
                chunk_result["chunk"] = chunk_result.get("chunk", "")
                
        # 파이프라인 완료 메트릭 기록
        team5_pipeline_duration.labels(service=service_name).observe(api_total_elapsed_time)

        if pipeline_error:
            logger.error(f"파이프라인 내에서 오류 발생: {pipeline_error}")
            # 파이프라인 레벨 에러 메트릭
            team5_pipeline_errors.labels(service=service_name, stage="pipeline", error_type="execution_error").inc()
            return {
                "result": api_result, # 오류가 발생했더라도 부분적인 결과가 있을 수 있음
                "error": pipeline_error,
                "total_elapsed_time": round(api_total_elapsed_time, 2)
            }

        logger.info(f"최종 API 반환 결과: {api_result}, 실행 시간: {api_total_elapsed_time:.2f}초")
        return {
            "result": api_result,
            "total_elapsed_time": round(api_total_elapsed_time, 2)
        }
            
    except Exception as e:
        # 이 블록은 invoke 자체의 실패 또는 그 이전 단계의 예외를 처리
        logger.error(f"run_pipeline 함수 실행 중 심각한 오류 발생: {str(e)}", exc_info=True)
        elapsed_time_on_error = round(time.time() - start_time, 2)
        
        # 파이프라인 레벨 에러 메트릭
        error_type = type(e).__name__
        team5_pipeline_errors.labels(service=service_name, stage="pipeline", error_type=error_type).inc()
        team5_pipeline_duration.labels(service=service_name).observe(elapsed_time_on_error)
        
        return {
            "result": [],
            "error": f"RAG 파이프라인 실행 중 심각한 오류: {str(e)}",
            "total_elapsed_time": elapsed_time_on_error
        }
    finally:
        # 활성 파이프라인 수 감소
        team5_pipeline_active_executions.labels(service=service_name).dec()
