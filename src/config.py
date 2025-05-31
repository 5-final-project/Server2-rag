# src/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MAX_RETRIES_PER_CHUNK: int = 1  # 청크당 최대 재시도 횟수
    MAX_RETRIES: int = 2            # 문서 관련성 평가 재시도 횟수
    GEMINI_API_KEY: str
    VECTOR_API_URL: str
    CHUNK_SIZE: int = 512          # 토큰 기준 상한
    CHUNK_OVERLAP: int = 64
    MAX_RETRY: int = 2             # 재적합 루프 (삭제 예정, MAX_RETRIES로 통합)
    VECTOR_TOP_K: int = 5
    RELEVANT_K: int = 3
    SCORE_THRESHOLD: float = 0.5
    TIMEOUT_SEC: int = 300         # 전 파이프라인 5 분
    LLM_MODEL: str = "gemini-2.5-flash-preview-05-20"  # 사용할 LLM 모델
    
    # 랭그래프 관련 설정
    LANGRAPH_RECURSION_LIMIT: int = 300  # 랭그래프 재귀 호출 한계
    
    # Prometheus 메트릭 관련 설정
    ENABLE_METRICS: bool = True    # 메트릭 활성화 여부
    METRICS_PORT: int = 8125       # 메트릭 포트 (FastAPI 서버와 동일)
    SERVICE_NAME: str = "server2-rag"  # 서비스 이름 (메트릭 라벨용)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

@lru_cache
def get_settings() -> Settings:
    return Settings()