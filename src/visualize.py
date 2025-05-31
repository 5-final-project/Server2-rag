"""
랭그래프 가시화 모듈 - 워크플로우 시각화 및 디버깅 기능 제공
"""
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
import networkx as nx
import matplotlib.pyplot as plt
from langgraph.checkpoint.memory import MemorySaver
from io import BytesIO
import base64
from pydantic import BaseModel

from src.graph import rag_workflow

logger = logging.getLogger("server2_rag")

class VisualizeOptions(BaseModel):
    """랭그래프 시각화 옵션"""
    figsize: tuple = (12, 8)
    title: str = "RAG Workflow Graph"
    node_size: int = 2000
    font_size: int = 10
    font_weight: str = "bold"
    edge_color: str = "gray"
    node_color: str = "#66c2a5"
    current_node_color: str = "#fc8d62"
    save_path: Optional[str] = None
    show_legend: bool = True
    layout: str = "spring"  # "spring", "circular", "kamada_kawai", "planar"
    dpi: int = 100

def get_langgraph_visualization(
    state_data: Optional[Dict[str, Any]] = None, 
    options: Optional[VisualizeOptions] = None,
    format: str = "base64"  # "base64", "file", "plt"
) -> Union[str, plt.Figure]:
    """
    랭그래프 워크플로우 시각화 생성
    
    Args:
        state_data: 현재 그래프 상태 데이터 (없으면 빈 그래프 생성)
        options: 시각화 옵션
        format: 반환 형식 (base64, file, plt)
        
    Returns:
        str 또는 plt.Figure: 시각화 결과 (형식에 따라 다름)
    """
    if options is None:
        options = VisualizeOptions()
    
    # 그래프 구조 추출
    compiled_graph = rag_workflow._graph
    nx_graph = compiled_graph.graph
    
    # 워크플로우 시각화
    plt.figure(figsize=options.figsize, dpi=options.dpi)
    
    # 레이아웃 선택
    if options.layout == "spring":
        pos = nx.spring_layout(nx_graph)
    elif options.layout == "circular":
        pos = nx.circular_layout(nx_graph)
    elif options.layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(nx_graph)
    elif options.layout == "planar":
        try:
            pos = nx.planar_layout(nx_graph)
        except:
            pos = nx.spring_layout(nx_graph)
    else:
        pos = nx.spring_layout(nx_graph)
    
    # 노드 색상 결정 (현재 활성 노드 강조)
    node_colors = []
    current_node = None
    
    if state_data:
        # 체크포인트에서 현재 노드 확인
        current_node = state_data.get("__langgraph_checkpoint__", {}).get("current_node")
    
    for node in nx_graph.nodes():
        if node == current_node:
            node_colors.append(options.current_node_color)
        else:
            node_colors.append(options.node_color)
    
    # 노드 그리기
    nx.draw_networkx_nodes(
        nx_graph, pos, 
        node_size=options.node_size,
        node_color=node_colors
    )
    
    # 엣지 그리기
    nx.draw_networkx_edges(
        nx_graph, pos, 
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        edge_color=options.edge_color,
        width=1.5
    )
    
    # 노드 라벨 그리기
    nx.draw_networkx_labels(
        nx_graph, pos,
        font_size=options.font_size,
        font_weight=options.font_weight
    )
    
    # 그래프 경로 정보 추가
    if state_data and "__langgraph_checkpoint__" in state_data:
        checkpoint = state_data["__langgraph_checkpoint__"]
        if "thread_id" in checkpoint and "path" in checkpoint:
            path_info = checkpoint["path"]
            if path_info and len(path_info) > 1:
                # 경로에 따라 엣지 강조
                path_edges = [(path_info[i], path_info[i+1]) for i in range(len(path_info)-1)]
                nx.draw_networkx_edges(
                    nx_graph, pos,
                    edgelist=path_edges,
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=15,
                    edge_color='red',
                    width=2.5
                )
    
    plt.title(options.title)
    plt.axis('off')
    
    # 범례 추가
    if options.show_legend:
        import matplotlib.patches as mpatches
        node_patch = mpatches.Patch(color=options.node_color, label='노드')
        current_node_patch = mpatches.Patch(color=options.current_node_color, label='현재 노드')
        path_line = mpatches.Patch(color='red', label='실행 경로')
        plt.legend(handles=[node_patch, current_node_patch, path_line], loc='upper right')
    
    # 반환 형식에 따라 처리
    if format == "file" and options.save_path:
        plt.savefig(options.save_path)
        plt.close()
        return options.save_path
    elif format == "base64":
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    else:  # "plt"
        return plt.gcf()

def get_graph_stats() -> Dict[str, Any]:
    """
    랭그래프 통계 정보 수집
    
    Returns:
        Dict[str, Any]: 그래프 통계 정보
    """
    compiled_graph = rag_workflow._graph
    nx_graph = compiled_graph.graph
    
    stats = {
        "node_count": nx_graph.number_of_nodes(),
        "edge_count": nx_graph.number_of_edges(),
        "nodes": list(nx_graph.nodes()),
        "edges": list(nx_graph.edges()),
        "conditional_edges": {},
        "entry_point": compiled_graph.entry_point
    }
    
    # 조건부 엣지 정보 수집
    if hasattr(compiled_graph, "conditional_edges"):
        for node, cond_fn in compiled_graph.conditional_edges.items():
            if hasattr(cond_fn, "__name__"):
                stats["conditional_edges"][node] = cond_fn.__name__
            else:
                stats["conditional_edges"][node] = str(cond_fn)
    
    return stats

def export_graph_as_json(save_path: str = "graph_definition.json") -> str:
    """
    그래프 정의를 JSON으로 내보내기
    
    Args:
        save_path: 저장할 파일 경로
        
    Returns:
        str: 저장된 파일 경로
    """
    stats = get_graph_stats()
    
    # 직렬화 가능한 형식으로 변환
    serializable_stats = {
        "node_count": stats["node_count"],
        "edge_count": stats["edge_count"],
        "nodes": [str(n) for n in stats["nodes"]],
        "edges": [[str(e[0]), str(e[1])] for e in stats["edges"]],
        "conditional_edges": {str(k): str(v) for k, v in stats["conditional_edges"].items()},
        "entry_point": str(stats["entry_point"])
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    
    return save_path

def generate_mermaid_diagram(state_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Mermaid.js 다이어그램 생성 (웹에서 표시 가능)
    
    Args:
        state_data: 현재 그래프 상태 (없으면 기본 그래프만 생성)
        
    Returns:
        str: Mermaid.js 다이어그램 문자열
    """
    compiled_graph = rag_workflow._graph
    nx_graph = compiled_graph.graph
    
    mermaid = ["```mermaid", "graph TD;"]
    
    # 현재 노드
    current_node = None
    if state_data and "__langgraph_checkpoint__" in state_data:
        current_node = state_data["__langgraph_checkpoint__"].get("current_node")
    
    # 노드 정의
    for node in nx_graph.nodes():
        node_str = str(node)
        if node == current_node:
            mermaid.append(f'    {node_str}["{node_str}"]:::currentNode;')
        else:
            mermaid.append(f'    {node_str}["{node_str}"];')
    
    # 엣지 정의
    for edge in nx_graph.edges():
        source, target = str(edge[0]), str(edge[1])
        
        # 실행 경로 표시
        is_path_edge = False
        if state_data and "__langgraph_checkpoint__" in state_data:
            path = state_data["__langgraph_checkpoint__"].get("path", [])
            if len(path) > 1:
                path_edges = [(str(path[i]), str(path[i+1])) for i in range(len(path)-1)]
                if (source, target) in path_edges:
                    is_path_edge = True
        
        if is_path_edge:
            mermaid.append(f'    {source} -->|"실행됨"| {target};')
        else:
            mermaid.append(f'    {source} --> {target};')
    
    # 스타일 정의
    mermaid.append("    classDef currentNode fill:#fc8d62,stroke:#333,stroke-width:2px;")
    mermaid.append("```")
    
    return "\n".join(mermaid)

def generate_workflow_trace(thread_id: str) -> Dict[str, Any]:
    """
    지정된 thread_id에 대한 워크플로우 실행 추적 정보 생성
    
    Args:
        thread_id: 추적할 워크플로우 스레드 ID
        
    Returns:
        Dict[str, Any]: 워크플로우 실행 추적 정보
    """
    if not hasattr(rag_workflow, "_checkpointer") or not isinstance(rag_workflow._checkpointer, MemorySaver):
        return {"error": "워크플로우가 MemorySaver를 사용하지 않습니다."}
    
    # MemorySaver에서 스레드 데이터 검색
    checkpointer = rag_workflow._checkpointer
    all_thread_data = checkpointer.checkpoints
    
    if thread_id not in all_thread_data:
        return {"error": f"스레드 ID '{thread_id}'에 대한 데이터를 찾을 수 없습니다."}
    
    thread_data = all_thread_data[thread_id]
    
    # 각 단계별 정보 수집
    steps = []
    
    for i, (checkpoint_id, checkpoint) in enumerate(thread_data.items()):
        if "__langgraph_checkpoint__" not in checkpoint:
            continue
        
        checkpoint_meta = checkpoint["__langgraph_checkpoint__"]
        current_node = checkpoint_meta.get("current_node", "unknown")
        path = checkpoint_meta.get("path", [])
        
        # 현재 단계 정보
        step_info = {
            "step": i,
            "checkpoint_id": checkpoint_id,
            "node": current_node,
            "path": [str(p) for p in path],
            "timestamp": checkpoint_meta.get("timestamp", 0),
        }
        
        # 노드별 주요 데이터 추출
        if current_node == "create_chunks":
            step_info["data"] = {
                "chunks_count": len(checkpoint.get("chunks", [])),
                "total_chunks": checkpoint.get("total_chunks", 0),
                "error": checkpoint.get("error")
            }
        elif current_node == "process_chunk":
            step_info["data"] = {
                "current_chunk_index": checkpoint.get("current_chunk_index", 0),
                "total_chunks": checkpoint.get("total_chunks", 0),
                "has_next": checkpoint.get("has_next", False),
                "error": checkpoint.get("error")
            }
        elif current_node == "generate_summary":
            step_info["data"] = {
                "chunk_length": len(checkpoint.get("chunk", "")),
                "error": checkpoint.get("error")
            }
        elif current_node == "decide_search":
            decision = checkpoint.get("decision", {})
            step_info["data"] = {
                "decision": decision.get("decision", False),
                "answer": decision.get("answer", ""),
                "success": decision.get("success", False),
                "error": checkpoint.get("error") or decision.get("error")
            }
        elif current_node == "search_documents":
            step_info["data"] = {
                "query": checkpoint.get("summary", {}).get("query", []),
                "similar_documents_count": len(checkpoint.get("similar_documents", [])),
                "error": checkpoint.get("error")
            }
        elif current_node == "evaluate_relevance":
            step_info["data"] = {
                "similar_documents_count": len(checkpoint.get("similar_documents", [])),
                "retry_count": checkpoint.get("retry_count", 0),
                "should_retry": checkpoint.get("should_retry", False),
                "feedback": checkpoint.get("feedback", ""),
                "error": checkpoint.get("error")
            }
        elif current_node == "update_results":
            step_info["data"] = {
                "results_count": len(checkpoint.get("results", [])),
                "current_chunk_index": checkpoint.get("current_chunk_index", 0),
                "total_chunks": checkpoint.get("total_chunks", 0),
                "has_next": checkpoint.get("has_next", False),
                "should_retry": checkpoint.get("should_retry", False),
                "error": checkpoint.get("error")
            }
        elif current_node == "finalize_results":
            final_result = checkpoint.get("final_result", {})
            step_info["data"] = {
                "result_count": len(final_result.get("result", [])),
                "total_elapsed_time": final_result.get("total_elapsed_time", 0),
                "error": final_result.get("error") or checkpoint.get("error")
            }
        
        steps.append(step_info)
    
    # 순서대로 정렬
    steps.sort(key=lambda x: x["step"])
    
    # 요약 정보 생성
    summary = {
        "thread_id": thread_id,
        "steps_count": len(steps),
        "start_time": steps[0]["timestamp"] if steps else 0,
        "end_time": steps[-1]["timestamp"] if steps else 0,
        "total_duration": (steps[-1]["timestamp"] - steps[0]["timestamp"]) if len(steps) > 1 else 0,
        "final_node": steps[-1]["node"] if steps else "unknown",
        "error": any(step.get("data", {}).get("error") for step in steps)
    }
    
    return {
        "summary": summary,
        "steps": steps
    }

def create_interactive_html(trace_data: Dict[str, Any], output_file: str = "workflow_trace.html") -> str:
    """
    워크플로우 추적 데이터를 기반으로 인터랙티브 HTML 생성
    
    Args:
        trace_data: generate_workflow_trace에서 생성된 추적 데이터
        output_file: 출력 HTML 파일 경로
        
    Returns:
        str: 생성된 HTML 파일 경로
    """
    summary = trace_data["summary"]
    steps = trace_data["steps"]
    
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='utf-8'>",
        "    <title>워크플로우 실행 추적</title>",
        "    <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css' rel='stylesheet'>",
        "    <script src='https://cdn.jsdelivr.net/npm/mermaid@9.4.3/dist/mermaid.min.js'></script>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; padding: 20px; }",
        "        .node-success { background-color: #d4edda; }",
        "        .node-error { background-color: #f8d7da; }",
        "        .step-details { display: none; margin-top: 10px; }",
        "        .accordion-button { cursor: pointer; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        f"        <h1>워크플로우 실행 추적: {summary['thread_id']}</h1>",
        "        <div class='card mb-4'>",
        "            <div class='card-header bg-primary text-white'>요약 정보</div>",
        "            <div class='card-body'>",
        f"                <p><strong>총 단계:</strong> {summary['steps_count']}</p>",
        f"                <p><strong>시작 시간:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(summary['start_time']))}</p>",
        f"                <p><strong>종료 시간:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(summary['end_time']))}</p>",
        f"                <p><strong>총 소요 시간:</strong> {summary['total_duration']:.2f}초</p>",
        f"                <p><strong>최종 노드:</strong> {summary['final_node']}</p>",
        f"                <p><strong>오류 발생:</strong> {'예' if summary['error'] else '아니오'}</p>",
        "            </div>",
        "        </div>",
        "        <h2>워크플로우 다이어그램</h2>",
        "        <div class='mermaid'>",
        "            graph TD;",
    ]
    
    # 그래프 노드 생성
    compiled_graph = rag_workflow._graph
    nx_graph = compiled_graph.graph
    
    # 마지막 단계의 현재 노드
    current_node = steps[-1]["node"] if steps else None
    
    # 실행된 경로 추적
    executed_paths = []
    for i in range(len(steps)-1):
        source = steps[i]["node"]
        target = steps[i+1]["node"]
        executed_paths.append((source, target))
    
    # 노드 정의
    for node in nx_graph.nodes():
        node_str = str(node)
        if node_str == current_node:
            html.append(f'            {node_str}["{node_str}"]:::currentNode;')
        elif any(step["node"] == node_str for step in steps):
            html.append(f'            {node_str}["{node_str}"]:::executedNode;')
        else:
            html.append(f'            {node_str}["{node_str}"];')
    
    # 엣지 정의
    for edge in nx_graph.edges():
        source, target = str(edge[0]), str(edge[1])
        
        if (source, target) in executed_paths:
            html.append(f'            {source} -->|"실행됨"| {target};')
        else:
            html.append(f'            {source} --> {target};')
    
    # 스타일 정의
    html.extend([
        "            classDef currentNode fill:#fc8d62,stroke:#333,stroke-width:2px;",
        "            classDef executedNode fill:#66c2a5,stroke:#333;",
        "        </div>",
        "        <h2>실행 단계</h2>",
        "        <div class='accordion' id='stepsAccordion'>",
    ])
    
    # 각 단계 상세 정보
    for i, step in enumerate(steps):
        step_id = f"step-{i}"
        node = step["node"]
        error = step.get("data", {}).get("error")
        node_class = "node-error" if error else "node-success"
        
        html.extend([
            f"            <div class='card mb-2 {node_class}'>",
            f"                <div class='card-header' id='heading-{step_id}'>",
            f"                    <h5 class='mb-0'>",
            f"                        <button class='accordion-button collapsed' type='button' data-bs-toggle='collapse' data-bs-target='#collapse-{step_id}' aria-expanded='false' aria-controls='collapse-{step_id}'>",
            f"                            단계 {i+1}: {node}",
            f"                        </button>",
            f"                    </h5>",
            f"                </div>",
            f"                <div id='collapse-{step_id}' class='collapse' aria-labelledby='heading-{step_id}' data-bs-parent='#stepsAccordion'>",
            f"                    <div class='card-body'>",
            f"                        <p><strong>시간:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(step['timestamp']))}</p>",
            f"                        <p><strong>경로:</strong> {' -> '.join(step['path'])}</p>",
        ])
        
        # 데이터 출력
        if "data" in step:
            html.append("                        <h6>데이터:</h6>")
            html.append("                        <pre><code>")
            html.append(json.dumps(step["data"], indent=4, ensure_ascii=False))
            html.append("                        </code></pre>")
        
        html.extend([
            "                    </div>",
            "                </div>",
            "            </div>",
        ])
    
    html.extend([
        "        </div>",
        "    </div>",
        "    <script src='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'></script>",
        "    <script>",
        "        mermaid.initialize({startOnLoad:true});",
        "        document.addEventListener('DOMContentLoaded', function() {",
        "            const accordionButtons = document.querySelectorAll('.accordion-button');",
        "            accordionButtons.forEach(button => {",
        "                button.addEventListener('click', function() {",
        "                    const target = document.querySelector(this.getAttribute('data-bs-target'));",
        "                    target.classList.toggle('show');",
        "                    this.classList.toggle('collapsed');",
        "                });",
        "            });",
        "        });",
        "    </script>",
        "</body>",
        "</html>",
    ])
    
    # HTML 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    
    return output_file 