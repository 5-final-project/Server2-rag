# src/app.py에 랭그래프 가시화 엔드포인트 추가

# ... 기존 임포트 ...
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import time
import uuid
import os
import logging
from typing import Dict, Any, List, Optional
from prometheus_client import start_http_server, REGISTRY, generate_latest, CONTENT_TYPE_LATEST

# 로깅 설정
from src.logging_utils import setup_json_logger
setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")

# 설정 및 그래프 관련 모듈 임포트
from src.config import get_settings
from src.graph import run_pipeline
from src.metrics import get_gpu_metrics_with_retry

# 가시화 모듈 임포트
from src.visualize import (
    get_langgraph_visualization, 
    generate_mermaid_diagram, 
    generate_workflow_trace,
    create_interactive_html,
    export_graph_as_json,
    VisualizeOptions
)

# ... 기존 코드 ...

# 설정 가져오기
settings = get_settings()

# FastAPI 앱 생성
app = FastAPI(
    title="RAG LangGraph API",
    description="랭그래프 기반 RAG 파이프라인 API",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus 메트릭 엔드포인트
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

# 입력 모델 정의
class RAGInput(BaseModel):
    text: str = Field(..., description="처리할 텍스트")
    thread_id: Optional[str] = Field(None, description="대화 스레드 ID (없으면 자동 생성)")

# RAG API 엔드포인트
@app.post("/api/rag")
async def process_rag(input_data: RAGInput):
    """
    텍스트를 RAG 파이프라인으로 처리
    """
    thread_id = input_data.thread_id or str(uuid.uuid4())
    logger.info(f"RAG 요청 시작. Thread ID: {thread_id}, 텍스트 길이: {len(input_data.text)}")
    
    try:
        result = run_pipeline(input_data.text, thread_id)
        return {
            "thread_id": thread_id,
            "result": result.get("result", []),
            "total_elapsed_time": result.get("total_elapsed_time", 0),
            "error": result.get("error")
        }
    except Exception as e:
        logger.error(f"RAG 처리 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG 처리 실패: {str(e)}")

# GPU 메트릭 엔드포인트
@app.get("/api/gpu")
async def get_gpu_metrics():
    """GPU 상태 정보 반환"""
    try:
        gpu_metrics = get_gpu_metrics_with_retry()
        if gpu_metrics:
            return gpu_metrics
        return {"error": "GPU 메트릭을 가져올 수 없습니다."}
    except Exception as e:
        return {"error": f"GPU 메트릭 수집 오류: {str(e)}"}

# 가시화 관련 엔드포인트 추가
@app.get("/api/visualize/graph", response_class=HTMLResponse)
async def visualize_workflow_graph():
    """워크플로우 그래프를 시각화하여 HTML로 반환"""
    try:
        img_base64 = get_langgraph_visualization(format="base64")
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>워크플로우 그래프 시각화</title>
        </head>
        <body>
            <h1>RAG 워크플로우 그래프</h1>
            <img src="data:image/png;base64,{img_base64}" alt="Workflow Graph">
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"그래프 시각화 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"그래프 시각화 실패: {str(e)}")

@app.get("/api/visualize/mermaid")
async def get_mermaid_diagram():
    """Mermaid.js 다이어그램 생성"""
    try:
        mermaid_code = generate_mermaid_diagram()
        return {"mermaid_code": mermaid_code}
    except Exception as e:
        logger.error(f"Mermaid 다이어그램 생성 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mermaid 다이어그램 생성 실패: {str(e)}")

@app.get("/api/visualize/trace/{thread_id}")
async def get_workflow_trace(thread_id: str):
    """특정 스레드의 워크플로우 실행 추적 정보"""
    try:
        trace_data = generate_workflow_trace(thread_id)
        if "error" in trace_data:
            raise HTTPException(status_code=404, detail=trace_data["error"])
        return trace_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"워크플로우 추적 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"워크플로우 추적 실패: {str(e)}")

@app.get("/api/visualize/html/{thread_id}", response_class=FileResponse)
async def get_workflow_html(thread_id: str):
    """워크플로우 실행 추적 HTML 생성"""
    try:
        trace_data = generate_workflow_trace(thread_id)
        if "error" in trace_data:
            raise HTTPException(status_code=404, detail=trace_data["error"])
            
        output_file = f"workflow_trace_{thread_id}.html"
        html_path = create_interactive_html(trace_data, output_file)
        return FileResponse(html_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HTML 생성 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"HTML 생성 실패: {str(e)}")

@app.get("/api/visualize/graph_definition")
async def get_graph_definition():
    """워크플로우 그래프 정의 JSON 추출"""
    try:
        json_path = export_graph_as_json()
        with open(json_path, 'r', encoding='utf-8') as f:
            graph_definition = json.load(f)
        return graph_definition
    except Exception as e:
        logger.error(f"그래프 정의 추출 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"그래프 정의 추출 실패: {str(e)}")

@app.get("/api/metrics/performance")
async def get_performance_metrics():
    """성능 메트릭 데이터 반환"""
    try:
        # 레지스트리에서 메트릭 수집
        collected_metrics = {}
        
        for metric in REGISTRY.collect():
            metric_name = metric.name
            metric_data = {"help": metric.documentation, "type": metric.type, "samples": []}
            
            for sample in metric.samples:
                if metric_name.startswith("team5_"):
                    sample_data = {
                        "name": sample.name,
                        "labels": sample.labels,
                        "value": sample.value
                    }
                    metric_data["samples"].append(sample_data)
            
            if metric_data["samples"]:
                collected_metrics[metric_name] = metric_data
        
        # 메트릭 카테고리별로 분류
        categorized_metrics = {
            "rag": {},
            "llm": {},
            "vector_search": {},
            "pipeline": {},
            "node": {},
            "gpu": {}
        }
        
        for name, data in collected_metrics.items():
            if "rag_" in name:
                categorized_metrics["rag"][name] = data
            elif "llm_" in name:
                categorized_metrics["llm"][name] = data
            elif "vector_search" in name:
                categorized_metrics["vector_search"][name] = data
            elif "pipeline_" in name:
                categorized_metrics["pipeline"][name] = data
            elif "node_" in name or "doc_relevance" in name or "search_" in name:
                categorized_metrics["node"][name] = data
            elif "gpu_" in name:
                categorized_metrics["gpu"][name] = data
        
        return categorized_metrics
    except Exception as e:
        logger.error(f"메트릭 수집 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"메트릭 수집 실패: {str(e)}")

# 정적 파일 서비스 설정 (가시화 HTML/이미지 등)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 메인 실행 함수
def main():
    """애플리케이션 메인 실행 함수"""
    import uvicorn
    
    # 기본 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Prometheus 메트릭 서버 시작 (설정에서 활성화된 경우)
    if settings.ENABLE_METRICS:
        metrics_port = settings.METRICS_PORT
        logger.info(f"Prometheus 메트릭 서버 시작 (포트: {metrics_port})")
        start_http_server(metrics_port)
    
    # FastAPI 서버 실행
    port = int(os.getenv("PORT", 8125))
    logger.info(f"RAG LangGraph API 서버 시작 (포트: {port})")
    
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main() 