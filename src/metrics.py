"""
Server2-rag 전용 Prometheus 메트릭 정의 모듈
- 순환 임포트 방지를 위해 모든 메트릭을 여기에 정의
- 다른 모듈에서는 이 파일에서 import하여 사용
"""

# GPU 모니터링 추가
import pynvml
import time
import logging
from prometheus_client import Counter, Histogram, Gauge, Summary

logger = logging.getLogger(__name__)

# ─── Team5 Server2 RAG 메트릭 정의 ──────────────────────────────────────

# ─── 기본 RAG 관련 메트릭 ─────────────────────────────────────────────
team5_rag_requests = Counter('team5_rag_requests_total', 'Total RAG requests', ['service', 'endpoint'])
team5_rag_duration = Histogram('team5_rag_processing_seconds', 'RAG processing time', ['service', 'endpoint'])
team5_rag_errors = Counter('team5_rag_errors_total', 'Total RAG errors', ['service', 'endpoint', 'error_type'])
team5_rag_chunk_count = Histogram('team5_rag_chunks_processed', 'Number of chunks processed per request', ['service'])
team5_rag_active_requests = Gauge('team5_rag_active_requests', 'Currently active RAG requests', ['service'])

# ─── LLM 관련 메트릭 ─────────────────────────────────────────────────
team5_llm_calls = Counter('team5_llm_calls_total', 'Total LLM API calls', ['service', 'llm_type'])
team5_llm_duration = Histogram('team5_llm_call_seconds', 'LLM API call duration', ['service', 'llm_type'])
team5_llm_errors = Counter('team5_llm_errors_total', 'Total LLM API errors', ['service', 'llm_type', 'error_type'])
team5_llm_tokens = Histogram('team5_llm_tokens_processed', 'Tokens processed by LLM', ['service', 'llm_type', 'token_type'])

# ─── 벡터 검색 관련 메트릭 ────────────────────────────────────────────
team5_vector_searches = Counter('team5_vector_searches_total', 'Total vector searches', ['service'])
team5_vector_search_duration = Histogram('team5_vector_search_seconds', 'Vector search duration', ['service'])
team5_vector_search_errors = Counter('team5_vector_search_errors_total', 'Total vector search errors', ['service', 'error_type'])
team5_vector_search_results = Histogram('team5_vector_search_results_count', 'Number of results returned', ['service'])

# ─── 파이프라인 관련 메트릭 ────────────────────────────────────────────
team5_pipeline_executions = Counter('team5_pipeline_executions_total', 'Total pipeline executions', ['service'])
team5_pipeline_duration = Histogram('team5_pipeline_execution_seconds', 'Pipeline execution time', ['service'])
team5_pipeline_errors = Counter('team5_pipeline_errors_total', 'Total pipeline errors', ['service', 'stage', 'error_type'])
team5_pipeline_chunk_processing = Histogram('team5_pipeline_chunk_processing_seconds', 'Time to process each chunk', ['service'])
team5_pipeline_active_executions = Gauge('team5_pipeline_active_executions', 'Currently active pipeline executions', ['service'])

# ─── GPU 메트릭 (Server1-whisper와 동일) ────────────────────────────────
team5_gpu_utilization = Gauge('team5_gpu_utilization_percent', 'Team5 GPU utilization', ['service'])
team5_gpu_memory_used = Gauge('team5_gpu_memory_used_mb', 'Team5 GPU memory used', ['service'])
team5_gpu_memory_total = Gauge('team5_gpu_memory_total_mb', 'Team5 GPU memory total', ['service'])
team5_gpu_temperature = Gauge('team5_gpu_temperature_celsius', 'Team5 GPU temperature', ['service'])
team5_gpu_power_usage = Gauge('team5_gpu_power_usage_watts', 'Team5 GPU power usage', ['service'])

# ─── 단계별 세부 성능 모니터링 메트릭 (추가) ───────────────────────────────
# 각 노드/단계별 처리 시간
team5_node_duration = Histogram('team5_node_duration_seconds', 'Node execution time', 
                               ['service', 'node_name'])

# 각 단계별 성공/실패 카운터
team5_node_success = Counter('team5_node_success_total', 'Node execution success count', 
                            ['service', 'node_name'])
team5_node_failure = Counter('team5_node_failure_total', 'Node execution failure count', 
                            ['service', 'node_name'])

# 검색 성공률 관련 메트릭
team5_search_success_rate = Gauge('team5_search_success_rate', 
                                 'Percentage of searches that returned relevant documents', 
                                 ['service'])
team5_search_relevance_score = Histogram('team5_search_relevance_score', 
                                        'Average relevance score of returned documents', 
                                        ['service'])

# 문서 관련성 평가 메트릭
team5_doc_relevance_hits = Counter('team5_doc_relevance_hits_total', 
                                  'Number of documents evaluated as relevant', 
                                  ['service'])
team5_doc_relevance_misses = Counter('team5_doc_relevance_misses_total', 
                                    'Number of documents evaluated as non-relevant', 
                                    ['service'])
team5_doc_relevance_precision = Gauge('team5_doc_relevance_precision', 
                                     'Precision of document relevance evaluation (relevant/total)', 
                                     ['service'])

# 재시도 관련 메트릭
team5_retry_count = Counter('team5_retry_count_total', 
                           'Number of retries performed', 
                           ['service', 'stage'])
team5_max_retry_reached = Counter('team5_max_retry_reached_total', 
                                 'Number of times max retry limit was reached', 
                                 ['service', 'stage'])

# LangGraph 노드 전환 메트릭
team5_node_transitions = Counter('team5_node_transitions_total', 
                               'Transitions between nodes in the graph', 
                               ['service', 'from_node', 'to_node'])

# 노드 처리 백분위 메트릭 (p50, p90, p95, p99)
team5_node_percentiles = Summary('team5_node_percentiles_seconds', 
                                'Node execution time percentiles', 
                                ['service', 'node_name'])

# ─── GPU 모니터링 유틸리티 함수 ─────────────────────────────────────────
def get_gpu_metrics_with_retry(max_retries=3):
    """GPU 메트릭을 재시도 로직과 함께 안전하게 가져오기 (Server1-whisper와 동일)"""
    for attempt in range(max_retries):
        try:
            # 매번 새로 초기화 (이전 상태 영향 방지)
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU 상태 확인
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 추가 메트릭 수집
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0
                
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
            except:
                power = 0
            
            return {
                'utilization': util.gpu,
                'memory_used': mem.used / 1024 / 1024,  # bytes to MB
                'memory_total': mem.total / 1024 / 1024,  # bytes to MB
                'temperature': temp,
                'power_usage': power
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                # 재시도 전 잠시 대기
                time.sleep(0.1)
                continue
            else:
                # 모든 재시도 실패
                logger.warning(f"GPU 메트릭 수집 실패: {e}")
                return None
    
    return None

# 성능 모니터링 유틸리티 함수
def update_search_relevance_metrics(service_name, relevant_count, total_count):
    """검색 관련성 메트릭 업데이트"""
    if total_count > 0:
        # 관련 문서 비율 계산 (0-100%)
        relevance_rate = (relevant_count / total_count) * 100
        team5_search_success_rate.labels(service=service_name).set(relevance_rate)
        
        # 문서 관련성 정밀도
        team5_doc_relevance_precision.labels(service=service_name).set(relevant_count / total_count)
    
    # 관련 문서 및 비관련 문서 카운트
    team5_doc_relevance_hits.labels(service=service_name).inc(relevant_count)
    team5_doc_relevance_misses.labels(service=service_name).inc(total_count - relevant_count)

def track_node_transition(service_name, from_node, to_node):
    """노드 간 전환 추적"""
    team5_node_transitions.labels(service=service_name, from_node=from_node, to_node=to_node).inc()

def track_node_execution(service_name, node_name, duration, success=True):
    """노드 실행 추적"""
    # 노드 실행 시간 기록
    team5_node_duration.labels(service=service_name, node_name=node_name).observe(duration)
    team5_node_percentiles.labels(service=service_name, node_name=node_name).observe(duration)
    
    # 성공/실패 카운트
    if success:
        team5_node_success.labels(service=service_name, node_name=node_name).inc()
    else:
        team5_node_failure.labels(service=service_name, node_name=node_name).inc()

def track_retry(service_name, stage, is_max_reached=False):
    """재시도 추적"""
    team5_retry_count.labels(service=service_name, stage=stage).inc()
    if is_max_reached:
        team5_max_retry_reached.labels(service=service_name, stage=stage).inc()