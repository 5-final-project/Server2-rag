"""
FastAPI 엔드포인트: POST /process
- 입력: { "text": "회의 전체 텍스트" }
- 출력: { "result": [ ... chunk별 RAG 결과 ... ] }
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import logging
import sys # sys 모듈 임포트
from .graph import run_pipeline

# -- 로깅 설정 시작 --
# 이 설정은 애플리케이션 전체에 적용되며, 다른 모듈에서 logging.getLogger(__name__)를 통해
# 로거를 가져와 사용하면 동일한 설정을 공유합니다.
logging.basicConfig(
    level=logging.INFO,  # INFO 레벨 이상의 모든 로그를 처리
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", # 로그 형식 지정
    handlers=[
        logging.StreamHandler(sys.stdout)  # 로그를 표준 출력(콘솔)으로 보냄
    ],
    force=True # 이미 설정된 핸들러가 있더라도 이 설정을 강제로 적용 (Uvicorn 등에서 설정한 핸들러 덮어쓰기)
)
# -- 로깅 설정 끝 --

# api.py 모듈 자체의 로거 (필요시 사용)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 이 파일의 로그만 DEBUG로 보고 싶다면 주석 해제

app = FastAPI()

class ProcessRequest(BaseModel):
    text: str

@app.post("/process")
async def process(req: ProcessRequest) -> Dict[str, Any]:
    logger.info(f"[API /process] 요청 수신. 텍스트 길이: {len(req.text)}자")
    if not req.text.strip():
        logger.warning("[API /process] 비어 있는 텍스트로 요청됨.")
        raise HTTPException(status_code=400, detail="text 필드는 비어 있을 수 없습니다.")
    
    logger.info("[API /process] run_pipeline 호출 시작...")
    pipeline_result = run_pipeline(req.text)
    logger.info(f"[API /process] run_pipeline 호출 완료. 결과: {pipeline_result}")
    return pipeline_result
