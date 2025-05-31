# src/llm.py
import logging
import re
import uuid
import time
from .logging_utils import setup_json_logger
from src.config import get_settings

# Prometheus 메트릭 임포트
from prometheus_client import Counter, Histogram

setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")

"""
LLM 호출 모듈
- summary: Gemini 1.5 Flash (raw text)
- decide/relevance: 설정에서 지정된 모델 (chat-style with Thought/Answer)
- 전역 backoff 재시도 로직
"""

import os
import logging
import time
from typing import Dict, Any
import google.generativeai as genai

from .metrics import (
    team5_llm_calls, 
    team5_llm_duration, 
    team5_llm_errors, 
    team5_llm_tokens
)

# 설정 가져오기
settings = get_settings()

# API 키 설정
API_KEY = settings.GEMINI_API_KEY
if API_KEY:
    genai.configure(api_key=API_KEY)
    logger.info(f"GEMINI_API_KEY loaded successfully (first 5 chars: {API_KEY[:5]}).")
else:
    logger.error("GEMINI_API_KEY not found in environment variables.")

MODEL_SUMMARY = "gemini-1.5-flash"  # 요약용 모델은 고정
MODEL_DECIDE = settings.LLM_MODEL   # 결정/평가용 모델은 설정에서 가져옴

# 요약 전용 시스템 프롬프트
SUMMARY_SYSTEM = (
    """당신은 회의 내용을 명확하고 검색하기 쉬운 형태로 요약하는 전문가입니다. 
    
다음 지침을 따라 주세요:
1. 회의 청크를 분석하여 주요 주제, 결정사항, 액션 아이템을 파악하세요.
2. 핵심 정보를 포함하는 2-3개의 완전한 문장으로 요약하세요. 단어 나열이 아닌 문장 형태로 작성하세요.
3. 엔티티, 관계, 구체적인 세부 사항 등 검색에 유용한 정보에 중점을 두세요.
4. 사용자가 이 정보를 검색할 때 사용할 만한 자연스러운 언어를 사용하세요.
5. 간결하면서도 정보가 풍부하게 작성하세요(60-80단어 이내).
6. 입력이 한국어인 경우 한국어로, 그렇지 않으면 영어로 작성하세요.
7. 중요한 이름, 날짜, 수치 등의 구체적인 정보를 포함하세요.

회의 청크:"""
)

# 검색 필요성 판단을 위한 시스템 프롬프트
DECIDE_SYSTEM = """
당신은 회의 내용을 분석하여 사내 문서 검색이 필요한지 판단하는 전문가입니다.

[지시사항]
1. 주어진 회의 청크와 요약을 주의 깊게 검토하세요.
2. 'Thought:'로 시작하는 줄에 상세한 판단 근거를 기술하세요.
3. 'Answer:'로 시작하는 줄에 최종 판단(Yes/No)을 기재하세요.

[검색이 필요한 경우 예시]
- 프로젝트 관련 구체적인 정보나 결정사항
- 기술적 용어, 제품명, 모듈명 등
- 회의 중 언급된 참고 문서나 자료
- 특정 주제에 대한 상세 설명이 필요한 내용
- 액션 아이템이나 후속 조치 사항

[검색이 필요 없는 경우 예시]
- 인사말 (예: "안녕하세요", "회의 시작하겠습니다")
- 일상적인 대화 (예: "커피 한 잔 하실래요?")
- 맥락 없는 단순한 의견 (예: "네, 그렇겠네요")
- 이미 완료된 일상 업무 보고

[판단 기준]
- 구체적이고 검색 가능한 정보가 포함되어 있으면 'Yes'
- 일상적이거나 맥락 없는 내용이면 'No'
- 불확실한 경우에도 'No'로 응답

[회의 청크]
{chunk}

[회의 요약]
{summary}"""

def _backoff_call(func, *args, retry=3, backoff_factor=1.0, **kwargs):
    """단일 backoff 재시도 공통 로직."""
    service_name = "server2-rag"
    
    # 메트릭 기록용으로 llm_type 추출하고 kwargs에서 제거
    llm_type = kwargs.pop('llm_type', 'unknown') if 'llm_type' in kwargs else 'unknown'
    
    for i in range(retry):
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # 성공 메트릭 기록
            team5_llm_calls.labels(service=service_name, llm_type=llm_type).inc()
            team5_llm_duration.labels(service=service_name, llm_type=llm_type).observe(duration)
            
            return result
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            
            # 에러 메트릭 기록
            error_type = type(e).__name__
            team5_llm_errors.labels(service=service_name, llm_type=llm_type, error_type=error_type).inc()
            
            if i < retry - 1:
                wait = backoff_factor * (2 ** i)
                logger.warning(f"Gemini call failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Gemini call failed after {retry} retries: {e}")
                raise
    
    raise RuntimeError("Gemini call failed after retries")

def generate_summary(chunk: str) -> str:
    trace_id = str(uuid.uuid4())
    logger.info({
        "event": "llm_summary_start",
        "trace_id": trace_id,
        "chunk_length": len(chunk),
        "chunk_preview": chunk[:100] + ("..." if len(chunk) > 100 else ""),
        "api_key_set": bool(API_KEY)  # API 키 설정 여부 로깅
    })
    
    """
    텍스트 청크에서 문장 형태의 요약 생성
    """
    if not API_KEY:
        error_msg = "Gemini API key is not configured."
        logger.error({
            "event": "llm_summary_error",
            "trace_id": trace_id,
            "error": error_msg
        })
        # API 키가 없을 때 예외 발생
        raise ValueError(error_msg)
    
    model = genai.GenerativeModel(MODEL_SUMMARY)
    prompt = SUMMARY_SYSTEM + "\n\n" + chunk
    
    try:
        logger.info({
            "event": "llm_summary_api_call",
            "trace_id": trace_id,
            "model": MODEL_SUMMARY,
            "prompt_length": len(prompt)
        })
        
        response = _backoff_call(model.generate_content, prompt, llm_type="gemini-1.5-flash")
        
        # 토큰 수 추정 및 메트릭 기록
        input_tokens = len(chunk.split())
        output_tokens = len(response.text.split()) if response.text else 0
        
        team5_llm_tokens.labels(service="server2-rag", llm_type="gemini-1.5-flash", token_type="input").observe(input_tokens)
        team5_llm_tokens.labels(service="server2-rag", llm_type="gemini-1.5-flash", token_type="output").observe(output_tokens)
        
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            logger.info({
                "event": "llm_summary_feedback",
                "trace_id": trace_id,
                "prompt_feedback": str(response.prompt_feedback)
            })
        
        # Check for blocking reasons specifically if text is empty
        if not response.text.strip() and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            for rating in response.prompt_feedback.safety_ratings:
                if rating.blocked:
                    block_msg = f"Content blocked by safety filter: {rating.category}"
                    logger.warning({
                        "event": "llm_summary_blocked",
                        "trace_id": trace_id,
                        "category": str(rating.category),
                        "confidence": str(rating.probability)
                    })
                    # 컨텐츠가 차단된 경우 예외 발생
                    raise ValueError(block_msg)
        
        summary = response.text.strip()
        
        # 빈 요약이 반환된 경우 예외 발생
        if not summary:
            empty_msg = "LLM returned empty summary"
            logger.warning({
                "event": "llm_summary_empty",
                "trace_id": trace_id,
                "message": empty_msg
            })
            raise ValueError(empty_msg)
        
        logger.info({
            "event": "llm_summary_success",
            "trace_id": trace_id,
            "summary_length": len(summary),
            "summary": summary
        })
        
        return summary
    except Exception as e:
        logger.error({
            "event": "llm_summary_error",
            "trace_id": trace_id,
            "error": str(e),
            "error_type": type(e).__name__
        }, exc_info=True)
        
        # 예외를 그대로 전파
        raise

def call_gemini_chat(prompt: str, system_prompt: str, model_name: str) -> dict:
    logger.info(f"[llm.call_gemini_chat] 호출됨. Model: {model_name}, System Prompt 길이: {len(system_prompt)}, Prompt 길이: {len(prompt)}")
    logger.info(f"[llm.call_gemini_chat] Attempting to use configured API_KEY (first 5 chars if set): {API_KEY[:5] if API_KEY else 'Not set'}")
    """
    chat-style 호출: Thought/Answer parsing
    """
    full_prompt = system_prompt + "\n\n" + prompt
    model = genai.GenerativeModel(model_name)
    
    # 모델 타입 결정
    llm_type = "gemini-1.5-flash" if "flash" in model_name.lower() else "gemini-2.5-pro"
    
    response = _backoff_call(model.generate_content, full_prompt, llm_type=llm_type)
    raw_text = response.text
    
    # 토큰 수 추정 및 메트릭 기록
    input_tokens = len(full_prompt.split())
    output_tokens = len(raw_text.split()) if raw_text else 0
    
    team5_llm_tokens.labels(service="server2-rag", llm_type=llm_type, token_type="input").observe(input_tokens)
    team5_llm_tokens.labels(service="server2-rag", llm_type=llm_type, token_type="output").observe(output_tokens)
    
    logger.info(f"[{model_name}] LLM Raw Response (before strip): {raw_text!r}")
    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        logger.info(f"[{model_name}] LLM prompt_feedback: {response.prompt_feedback}")
    else:
        logger.info(f"[{model_name}] No prompt_feedback attribute or it's empty.")

    # Check for blocking reasons specifically if text is empty
    if not raw_text.strip() and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        for rating in response.prompt_feedback.safety_ratings:
            if rating.blocked:
                logger.warning(f"[{model_name}] Response was blocked due to: {rating.category}. Confidence: {rating.probability}")
                return {"thought": "Content blocked by safety filter", "answer": "Error: Content blocked"}

    text = raw_text.strip()
    thought, answer, feedback = "", "", ""

    # Thought 추출 (Answer: 전까지 또는 Feedback: 전까지 또는 문자열 끝까지)
    thought_match = re.search(r"Thought:(.*?)(Answer:|Feedback:|$)", text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Answer 추출 (Feedback: 전까지 또는 문자열 끝까지)
    answer_match = re.search(r"Answer:(.*?)(Feedback:|$)", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()

    # Feedback 추출 (문자열 끝까지)
    feedback_match = re.search(r"Feedback:(.*)", text, re.DOTALL | re.IGNORECASE)
    if feedback_match:
        feedback = feedback_match.group(1).strip()

    # 만약 태그 없이 일반 텍스트만 반환된 경우, 이를 answer로 간주할지 여부
    if not thought_match and not answer_match and not feedback_match and text:
        logger.warning(f"[{model_name}] LLM response did not contain expected Thought/Answer/Feedback tags. Full text assigned to answer: {text!r}")
        answer = text

    return {"thought": thought, "answer": answer, "feedback": feedback}

def decide_need_search(chunk: str, summary: str) -> dict:
    logger.info(f"[llm.decide_need_search] 호출됨. Chunk 길이: {len(chunk)}, Summary: {summary}")
    """
    주어진 회의 청크와 요약을 바탕으로 사내 문서 검색이 필요한지 판단합니다.
    
    Args:
        chunk: 분석할 회의 청크 텍스트
        summary: 해당 청크의 요약 정보
        
    Returns:
        dict: {
            "thought": 판단 근거 (LLM의 생각),
            "answer": "Yes" 또는 "No" (LLM의 답변),
            "decision": boolean (True if "Yes"),
            "success": boolean (작업 성공 여부),
            "error": str | None (오류 메시지)
        }
    """
    if not API_KEY:
        logger.error("[llm.decide_need_search] GEMINI_API_KEY is not set. Cannot call LLM.")
        return {
            "thought": "GEMINI_API_KEY is not set. Cannot call LLM.",
            "answer": "No",
            "decision": False,
            "success": False,
            "error": "API key not configured"
        }

    system_prompt_formatted = DECIDE_SYSTEM.format(chunk=chunk, summary=summary)
    
    try:
        llm_response = call_gemini_chat(prompt="", system_prompt=system_prompt_formatted, model_name=MODEL_DECIDE)
        
        llm_thought = llm_response.get("thought", "")
        llm_answer_original = llm_response.get("answer", "")
        
        normalized_answer = llm_answer_original.strip().lower()
        calculated_decision = normalized_answer == 'yes'
        
        logger.info(f"[llm.decide_need_search] LLM Thought: '{llm_thought}'")
        logger.info(f"[llm.decide_need_search] LLM Answer (Original): '{llm_answer_original}', Normalized: '{normalized_answer}', Calculated Decision: {calculated_decision}")

        return {
            "thought": llm_thought,
            "answer": llm_answer_original,
            "decision": calculated_decision,
            "success": True,
            "error": None
        }
    except Exception as e:
        logger.error(f"[llm.decide_need_search] Error during LLM call or processing: {e}", exc_info=True)
        return {
            "thought": f"Error processing LLM response: {e}",
            "answer": "No",
            "decision": False,
            "success": False,
            "error": str(e)
        }

def decide_relevance(chunk: str, summary: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"[llm.decide_relevance] 호출됨. Chunk 길이: {len(chunk)}, Summary: {summary}, Doc Keys: {list(doc.keys())}")
    """
    LLM을 사용하여 문서의 관련성을 평가하고, 재시도 필요 여부 및 피드백을 반환합니다.

    Args:
        chunk: 원본 청크 텍스트
        summary: 청크의 요약
        doc: 평가할 문서 (page_content와 metadata 포함)

    Returns:
        Dict[str, Any]: {
            "thought": str,        # LLM의 분석/판단 과정
            "answer": str,         # LLM의 원본 답변 문자열 (Yes/No 등)
            "relevant": bool,      # 최종 관련성 여부 (True/False)
            "feedback": str,       # 재시도 시 개선을 위한 LLM의 피드백
            "retry_needed": bool,  # 재시도 필요 여부
            "success": bool,       # LLM 호출 및 처리 성공 여부
            "error": Optional[str] # 오류 메시지 (실패 시)
        }
    """
    relevance_system = """당신은 문서 관련성 평가 전문가입니다. 다음 지침에 따라 분석을 수행해주세요:

1. 'Thought:'로 시작하는 줄에 상세한 분석 과정을 기술하세요.
2. 'Answer:'로 시작하는 줄에 최종 판단(Yes/No)을 기재하세요.
3. 'Feedback:'으로 시작하는 줄에 검색 쿼리 개선을 위한 제안을 해주세요 (문서가 부적절하다고 판단될 경우).

[평가 기준]
- 문서가 회의 요약의 맥락과 일치하는가?
- 문서가 회의에서 언급된 구체적인 정보를 뒷받침하는가?
- 문서가 회의의 결정 사항이나 액션 아이템과 관련이 있는가?
- 문서가 회의의 핵심 주제를 다루고 있는가?

[출력 형식]
Thought: [분석 과정]
Answer: [Yes/No]
Feedback: [검색 쿼리 개선을 위한 제안]

---
[회의 청크]
{chunk}

[회의 요약]
{summary}

[문서 제목]
{title}

[문서 내용]
{content}
---
"""
    try:
        # 문서에서 제목과 내용 추출
        doc_content = doc.get("page_content", "")
        doc_metadata = doc.get("metadata", {})
        doc_title = doc_metadata.get("title", "제목 없음")

        prompt = relevance_system.format(
            chunk=chunk,
            summary=summary,
            title=doc_title,
            content=doc_content[:2000]  # 내용이 너무 길 경우 잘라내기
        )

        # call_gemini_chat은 이제 {'thought': ..., 'answer': ..., 'feedback': ...} 반환
        llm_response = call_gemini_chat(prompt, "", MODEL_DECIDE)

        thought = llm_response.get("thought", "")
        # LLM의 Yes/No 답변 원본 (파싱 전)
        llm_answer_original = llm_response.get("answer", "No") 
        feedback = llm_response.get("feedback", "")

        # 최종 관련성 판단 (boolean)
        normalized_answer = llm_answer_original.strip().lower()
        relevant = normalized_answer.startswith("yes")

        # 재시도 필요 여부 결정 (예: 관련 없고 피드백 있으면 재시도)
        retry_needed = not relevant and bool(feedback)
        
        logger.info(f"[llm.decide_relevance] LLM Thought: '{thought}'")
        logger.info(f"[llm.decide_relevance] LLM Answer (Original): '{llm_answer_original}', Normalized: '{normalized_answer}', Calculated Relevant: {relevant}")
        logger.info(f"[llm.decide_relevance] LLM Feedback: '{feedback}', Calculated Retry Needed: {retry_needed}")

        return {
            "thought": thought,
            "answer": llm_answer_original, # LLM의 원본 답변
            "relevant": relevant,
            "feedback": feedback,
            "retry_needed": retry_needed,
            "success": True,
            "error": None
        }
    except Exception as e:
        error_msg = f"Error during LLM call or processing in decide_relevance: {str(e)}"
        logger.error(f"[llm.decide_relevance] {error_msg}", exc_info=True)
        return {
            "thought": error_msg,
            "answer": "No", # 오류 시 기본값
            "relevant": False,
            "feedback": "",
            "retry_needed": True, # 오류 시 재시도 고려
            "success": False,
            "error": str(e)
        }